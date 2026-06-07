#include "pipeline.h"

#include "camera_capture.h"
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

    std::cout << "[Pipeline] Init complete\n";
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

    std::cout << "[Pipeline] Stopped\n";
    auto s = GetStats();
    std::cout << "  Total frames: " << s.total_frames << "\n";
    std::cout << "  Defect frames: " << s.defect_frames << "\n";
    std::cout << "  Avg capture: " << s.avg_capture_ms << " ms\n";
    std::cout << "  Avg preprocess: " << s.avg_preprocess_ms << " ms\n";
    std::cout << "  Avg infer: " << s.avg_infer_ms << " ms\n";
    std::cout << "  Avg postprocess: " << s.avg_postprocess_ms << " ms\n";
    std::cout << "  Avg E2E: " << s.avg_e2e_ms << " ms\n";
}

PipelineStats Pipeline::GetStats() const {
    return stats_;
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

        CudaPreprocessResize(
            frame->gpu_data, orig_w, orig_h,
            engine_->InputGpuPtr(),
            config_.trt.input_width, config_.trt.input_height,
            engine_->Stream());

        auto t_pre = std::chrono::high_resolution_clock::now();
        double ms_pre = std::chrono::duration<double, std::milli>(t_pre - t_cap).count();

        if (!engine_->Infer()) {
            ring_->ReleaseReadSlot(frame);
            continue;
        }

        auto t_infer = std::chrono::high_resolution_clock::now();
        double ms_infer = std::chrono::duration<double, std::milli>(t_infer - t_pre).count();

        int num_rows = engine_->OutputShape(1);
        int num_classes = engine_->OutputShape(2) - 4;
        if (num_rows <= 0) num_rows = 8400;
        if (num_classes <= 0) num_classes = 9;

        auto dets = DecodeYoloV8Output(
            engine_->OutputGpuPtr(),
            num_rows, num_classes,
            config_.trt.score_threshold,
            config_.trt.nms_threshold,
            params, 0);

        auto t_post = std::chrono::high_resolution_clock::now();
        double ms_post = std::chrono::duration<double, std::milli>(t_post - t_infer).count();

        double ms_e2e = std::chrono::duration<double, std::milli>(t_post - t_start).count();

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

            stats_.total_frames = count;
            stats_.defect_frames = defect_count;
            stats_.avg_capture_ms = sum_capture / count;
            stats_.avg_preprocess_ms = sum_pre / count;
            stats_.avg_infer_ms = sum_infer / count;
            stats_.avg_postprocess_ms = sum_post / count;
            stats_.avg_e2e_ms = sum_e2e / count;
        }

        if (config_.save_result_image && !dets.empty()) {
            cv::Mat img(orig_h, orig_w, CV_8UC1);
            cudaMemcpy(img.data, frame->gpu_data,
                orig_w * orig_h, cudaMemcpyDeviceToHost);
            cv::Mat color;
            cv::cvtColor(img, color, cv::COLOR_GRAY2BGR);
            DrawDetections(color, dets, class_names);

            std::ostringstream oss;
            oss << config_.output_dir << "frame_" << std::setw(6) << std::setfill('0')
                << frame->meta.frame_id << ".jpg";
            cv::imwrite(oss.str(), color);
        }

        if (frame->meta.frame_id % 100 == 0) {
            std::cout << "[Pipeline] Frame " << frame->meta.frame_id
                      << " | E2E=" << std::fixed << std::setprecision(2) << ms_e2e
                      << "ms (cap=" << ms_capture
                      << " pre=" << ms_pre
                      << " inf=" << ms_infer
                      << " post=" << ms_post
                      << ") | defects=" << dets.size() << "\n";
        }

        ring_->ReleaseReadSlot(frame);
    }
}

}
