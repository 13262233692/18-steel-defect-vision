#include "pipeline.h"

#include "camera_capture.h"
#include "defect_tracker.h"
#include "trt_engine.h"
#include "yolo_postprocess.h"

#include <opencv2/opencv.hpp>

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>

namespace steel {

Pipeline::Pipeline(const PipelineConfig& cfg) : config_(cfg) {}

Pipeline::~Pipeline() {
    Stop();

    if (preprocess_done_) cudaEventDestroy(preprocess_done_);
    if (infer_done_) cudaEventDestroy(infer_done_);
    if (preprocess_stream_) cudaStreamDestroy(preprocess_stream_);
    if (infer_stream_) cudaStreamDestroy(infer_stream_);
    if (postprocess_stream_) cudaStreamDestroy(postprocess_stream_);
    if (tracker_stream_) cudaStreamDestroy(tracker_stream_);
    if (prev_overlap_gpu_) cudaFree(prev_overlap_gpu_);
    if (curr_overlap_gpu_) cudaFree(curr_overlap_gpu_);
}

bool Pipeline::Init() {
    ring_ = std::make_unique<RingBuffer>(
        config_.ring_buffer_size,
        config_.camera.image_width,
        config_.camera.image_height);

    capture_ = std::make_unique<CameraCapture>(config_.camera, ring_.get());

    engine_ = std::make_unique<TrtEngine>(config_.trt);
    if (!engine_->BuildOrLoadEngine()) {
        std::cerr << "[Pipeline] Failed to init TRT engine\n";
        return false;
    }

    tracker_ = std::make_unique<DefectTracker>(config_.tracker);

    cudaError_t err;
    err = cudaStreamCreateWithFlags(&preprocess_stream_, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        std::cerr << "[Pipeline] Failed to create preprocess stream\n";
        return false;
    }
    err = cudaStreamCreateWithFlags(&infer_stream_, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        std::cerr << "[Pipeline] Failed to create infer stream\n";
        return false;
    }
    err = cudaStreamCreateWithFlags(&postprocess_stream_, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        std::cerr << "[Pipeline] Failed to create postprocess stream\n";
        return false;
    }
    err = cudaStreamCreateWithFlags(&tracker_stream_, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        std::cerr << "[Pipeline] Failed to create tracker stream\n";
        return false;
    }

    err = cudaEventCreateWithFlags(&preprocess_done_, cudaEventDisableTiming);
    if (err != cudaSuccess) {
        std::cerr << "[Pipeline] Failed to create preprocess_done event\n";
        return false;
    }
    err = cudaEventCreateWithFlags(&infer_done_, cudaEventDisableTiming);
    if (err != cudaSuccess) {
        std::cerr << "[Pipeline] Failed to create infer_done event\n";
        return false;
    }

    overlap_width_ = config_.camera.image_width;
    overlap_height_ = config_.tracker.overlap_rows;
    size_t overlap_bytes = static_cast<size_t>(overlap_width_) * overlap_height_;
    err = cudaMalloc(&prev_overlap_gpu_, overlap_bytes);
    if (err != cudaSuccess) {
        std::cerr << "[Pipeline] Failed to alloc prev overlap buffer\n";
        return false;
    }
    err = cudaMalloc(&curr_overlap_gpu_, overlap_bytes);
    if (err != cudaSuccess) {
        std::cerr << "[Pipeline] Failed to alloc curr overlap buffer\n";
        return false;
    }

    std::cout << "[Pipeline] Init complete (4 non-blocking streams + 2 sync events + tracker)\n";
    std::cout << "[Pipeline] Overlap region: " << overlap_width_ << "x" << overlap_height_ << "\n";
    return true;
}

bool Pipeline::Start() {
    if (running_.load()) return true;
    running_.store(true);

    if (!capture_->Start()) {
        std::cerr << "[Pipeline] Failed to start camera capture\n";
        running_.store(false);
        return false;
    }

    infer_thread_ = std::thread(&Pipeline::InferenceLoop, this);

    std::cout << "[Pipeline] Started\n";
    return true;
}

void Pipeline::Stop() {
    if (!running_.load()) return;
    running_.store(false);

    capture_->Stop();

    if (infer_thread_.joinable()) {
        infer_thread_.join();
    }

    if (tracker_) {
        tracker_->PrintStats();
        auto final_tracks = tracker_->FinalizeTracks();
        std::cout << "\n[Pipeline] ====== Defect Tracking Report ======\n";
        std::cout << "  Total unique defects: " << final_tracks.size() << "\n";
        for (const auto& t : final_tracks) {
            auto class_names = DefaultDefectClassNames();
            std::string cls = (t.class_id < static_cast<int>(class_names.size()))
                ? class_names[t.class_id] : "cls" + std::to_string(t.class_id);
            std::cout << "  #" << t.track_id << " " << cls
                      << " | length=" << std::fixed << std::setprecision(1) << t.total_length << "px"
                      << " | frames=" << t.frame_count
                      << " | conf=" << std::setprecision(2) << t.confidence
                      << " | span=[" << t.first_frame_id << "-" << t.last_frame_id << "]"
                      << "\n";
        }
        std::cout << "[Pipeline] ====================================\n";
    }

    std::cout << "[Pipeline] Stopped\n";
    auto s = GetStats();
    std::cout << "  Total frames: " << s.total_frames << "\n";
    std::cout << "  Defect frames: " << s.defect_frames << "\n";
    std::cout << "  Unique defects: " << s.total_unique_defects << "\n";
    std::cout << "  Avg capture: " << s.avg_capture_ms << " ms\n";
    std::cout << "  Avg preprocess: " << s.avg_preprocess_ms << " ms\n";
    std::cout << "  Avg infer: " << s.avg_infer_ms << " ms\n";
    std::cout << "  Avg postprocess: " << s.avg_postprocess_ms << " ms\n";
    std::cout << "  Avg E2E: " << s.avg_e2e_ms << " ms\n";
}

PipelineStats Pipeline::GetStats() const {
    return stats_;
}

void Pipeline::CacheOverlapRegion(const uint8_t* gpu_frame, int width, int height,
                                   cudaStream_t stream) {
    if (!gpu_frame || overlap_height_ > height) return;

    size_t overlap_bytes = static_cast<size_t>(width) * overlap_height_;

    if (has_prev_overlap_) {
        cudaMemcpyAsync(prev_overlap_gpu_, curr_overlap_gpu_, overlap_bytes,
            cudaMemcpyDeviceToDevice, stream);
    }

    const uint8_t* bottom_row = gpu_frame + static_cast<size_t>(width) * (height - overlap_height_);
    cudaMemcpyAsync(curr_overlap_gpu_, bottom_row, overlap_bytes,
        cudaMemcpyDeviceToDevice, stream);

    has_prev_overlap_ = true;
}

void Pipeline::InferenceLoop() {
    auto class_names = DefaultDefectClassNames();

    double sum_capture = 0, sum_pre = 0, sum_infer = 0, sum_post = 0, sum_e2e = 0;
    uint64_t count = 0;
    uint64_t defect_count = 0;

    int warmup = config_.warmup_iterations;

    while (running_.load()) {
        auto t_start = std::chrono::high_resolution_clock::now();

        Frame* frame = ring_->AcquireReadSlot();
        if (!frame || !frame->meta.valid) {
            ring_->ReleaseReadSlot(frame);
            continue;
        }

        auto t_cap = std::chrono::high_resolution_clock::now();
        double ms_capture = std::chrono::duration<double, std::milli>(t_cap - t_start).count();

        PreprocessParams params;
        int orig_w = frame->meta.width;
        int orig_h = frame->meta.height;
        float ratio = std::min(static_cast<float>(config_.trt.input_width) / orig_w,
                               static_cast<float>(config_.trt.input_height) / orig_h);
        int new_w = static_cast<int>(orig_w * ratio);
        int new_h = static_cast<int>(orig_h * ratio);
        params.pad_w = (config_.trt.input_width - new_w) / 2;
        params.pad_h = (config_.trt.input_height - new_h) / 2;
        params.scale_x = ratio;
        params.scale_y = ratio;
        params.orig_w = orig_w;
        params.orig_h = orig_h;

        cudaStreamWaitEvent(preprocess_stream_, frame->h2d_done, 0);

        CudaPreprocessResize(
            frame->gpu_data, orig_w, orig_h,
            engine_->InputGpuPtr(),
            config_.trt.input_width, config_.trt.input_height,
            preprocess_stream_);

        cudaEventRecord(preprocess_done_, preprocess_stream_);

        auto t_pre = std::chrono::high_resolution_clock::now();
        double ms_pre = std::chrono::duration<double, std::milli>(t_pre - t_cap).count();

        cudaStreamWaitEvent(infer_stream_, preprocess_done_, 0);

        if (!engine_->Infer(infer_stream_)) {
            ring_->ReleaseReadSlot(frame);
            continue;
        }

        cudaEventRecord(infer_done_, infer_stream_);

        auto t_infer = std::chrono::high_resolution_clock::now();
        double ms_infer = std::chrono::duration<double, std::milli>(t_infer - t_pre).count();

        cudaStreamWaitEvent(postprocess_stream_, infer_done_, 0);

        int num_rows = engine_->OutputShape(1);
        int num_classes = engine_->OutputShape(2) - 4;
        if (num_rows <= 0) num_rows = 8400;
        if (num_classes <= 0) num_classes = 9;

        auto dets = DecodeYoloV8Output(
            engine_->OutputGpuPtr(),
            num_rows, num_classes,
            config_.trt.score_threshold,
            config_.trt.nms_threshold,
            params, postprocess_stream_);

        cudaStreamSynchronize(postprocess_stream_);

        auto t_post = std::chrono::high_resolution_clock::now();
        double ms_post = std::chrono::duration<double, std::milli>(t_post - t_infer).count();

        double ms_e2e = std::chrono::duration<double, std::milli>(t_post - t_start).count();

        cudaStreamWaitEvent(tracker_stream_, frame->h2d_done, 0);

        CacheOverlapRegion(frame->gpu_data, orig_w, orig_h, tracker_stream_);

        if (!dets.empty()) {
            OverlapPatch prev_patch{}, curr_patch{};
            if (has_prev_overlap_) {
                prev_patch.gpu_data = prev_overlap_gpu_;
                prev_patch.width = overlap_width_;
                prev_patch.height = overlap_height_;
                prev_patch.stride = overlap_width_;
            }
            curr_patch.gpu_data = curr_overlap_gpu_;
            curr_patch.width = overlap_width_;
            curr_patch.height = overlap_height_;
            curr_patch.stride = overlap_width_;

            tracker_->Update(dets, prev_patch, curr_patch,
                           frame->meta.frame_id, tracker_stream_);
        }

        const auto& active_tracks = tracker_->ActiveTracks();
        for (auto& d : dets) {
            for (const auto& t : active_tracks) {
                float tcx = (t.x1 + t.x2) * 0.5f;
                float dcx = (d.x1 + d.x2) * 0.5f;
                float tcy = (t.y1 + t.y2) * 0.5f;
                float dcy = (d.y1 + d.y2) * 0.5f;
                float dist = sqrtf((tcx - dcx) * (tcx - dcx) + (tcy - dcy) * (tcy - dcy));
                float tsize = std::max((t.x2 - t.x1), (t.y2 - t.y1));
                if (dist < tsize * 0.8f && t.class_id == d.class_id) {
                    d.track_id = t.track_id;
                    break;
                }
            }
        }

        cudaStreamSynchronize(tracker_stream_);

        if (warmup > 0) {
            warmup--;
        } else {
            count++;
            sum_capture += ms_capture;
            sum_pre += ms_pre;
            sum_infer += ms_infer;
            sum_post += ms_post;
            sum_e2e += ms_e2e;

            if (!dets.empty()) defect_count++;

            int unique_count = 0;
            for (const auto& t : active_tracks) {
                if (t.active) unique_count++;
            }

            stats_.total_frames = count;
            stats_.defect_frames = defect_count;
            stats_.total_unique_defects = unique_count;
            stats_.avg_capture_ms = sum_capture / count;
            stats_.avg_preprocess_ms = sum_pre / count;
            stats_.avg_infer_ms = sum_infer / count;
            stats_.avg_postprocess_ms = sum_post / count;
            stats_.avg_e2e_ms = sum_e2e / count;
        }

        if (config_.save_result_image && !dets.empty()) {
            cv::Mat img(orig_h, orig_w, CV_8UC1);
            cudaMemcpyAsync(img.data, frame->gpu_data,
                orig_w * orig_h, cudaMemcpyDeviceToHost, tracker_stream_);
            cudaStreamSynchronize(tracker_stream_);
            cv::Mat color;
            cv::cvtColor(img, color, cv::COLOR_GRAY2BGR);
            DrawDetections(color, dets, class_names);

            std::ostringstream oss;
            oss << config_.output_dir << "frame_" << std::setw(6) << std::setfill('0')
                << frame->meta.frame_id << ".jpg";
            cv::imwrite(oss.str(), color);
        }

        if (frame->meta.frame_id % 100 == 0) {
            int unique_count = 0;
            for (const auto& t : active_tracks) {
                if (t.active) unique_count++;
            }
            std::cout << "[Pipeline] Frame " << frame->meta.frame_id
                      << " | E2E=" << std::fixed << std::setprecision(2) << ms_e2e
                      << "ms (cap=" << ms_capture
                      << " pre=" << ms_pre
                      << " inf=" << ms_infer
                      << " post=" << ms_post
                      << ") | dets=" << dets.size()
                      << " tracks=" << unique_count
                      << "\n";
        }

        ring_->ReleaseReadSlot(frame);
    }
}

}
