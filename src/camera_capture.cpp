#include "camera_capture.h"

#include <chrono>
#include <cstring>
#include <iostream>
#include <random>

#ifdef USE_BASLER
#include <pylc/GenApi/GenApi.h>
#include <pylc/PylonCamera.h>
#endif

#ifdef USE_HIKVISION
#include <MvCameraControl.h>
#endif

namespace steel {

RingBuffer::RingBuffer(int capacity, int width, int height)
    : capacity_(capacity), width_(width), height_(height) {
    frames_.resize(capacity);
    slot_state_.resize(capacity);
    for (int i = 0; i < capacity; ++i) {
        AllocPinnedFrameBuffer(width, height,
            &frames_[i].gpu_data, &frames_[i].host_pinned, &frames_[i].h2d_done);
        frames_[i].meta.width = width;
        frames_[i].meta.height = height;
        frames_[i].meta.valid = false;
        slot_state_[i].store(0);
    }
}

RingBuffer::~RingBuffer() {
    for (auto& f : frames_) {
        FreePinnedFrameBuffer(f.gpu_data, f.host_pinned, f.h2d_done);
    }
}

Frame* RingBuffer::AcquireWriteSlot() {
    std::unique_lock<std::mutex> lk(write_mtx_);
    write_cv_.wait(lk, [this]() {
        return slot_state_[write_idx_].load() == 0;
    });
    Frame* f = &frames_[write_idx_];
    slot_state_[write_idx_].store(1);
    return f;
}

void RingBuffer::ReleaseWriteSlot(Frame*) {
    int idx = write_idx_;
    slot_state_[idx].store(2);
    write_idx_ = (write_idx_ + 1) % capacity_;
    read_cv_.notify_one();
}

Frame* RingBuffer::AcquireReadSlot() {
    std::unique_lock<std::mutex> lk(read_mtx_);
    read_cv_.wait(lk, [this]() {
        return slot_state_[read_idx_].load() == 2;
    });
    Frame* f = &frames_[read_idx_];
    slot_state_[read_idx_].store(3);
    return f;
}

void RingBuffer::ReleaseReadSlot(Frame*) {
    int idx = read_idx_;
    slot_state_[idx].store(0);
    read_idx_ = (read_idx_ + 1) % capacity_;
    write_cv_.notify_one();
}

CameraCapture::CameraCapture(const CameraConfig& cfg, RingBuffer* ring)
    : config_(cfg), ring_(ring) {}

CameraCapture::~CameraCapture() {
    Stop();
    if (h2d_stream_) {
        cudaStreamDestroy(h2d_stream_);
        h2d_stream_ = nullptr;
    }
}

bool CameraCapture::Start() {
    if (running_.load()) return true;

    cudaError_t err = cudaStreamCreateWithFlags(&h2d_stream_, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        std::cerr << "[CameraCapture] Failed to create non-blocking H2D stream\n";
        return false;
    }

    running_.store(true);

    bool use_sim = true;

    if (InitBasler()) {
        use_sim = false;
    } else if (InitHikvision()) {
        use_sim = false;
    }

    if (use_sim) {
        thread_ = std::thread(&CameraCapture::SimulateThread, this);
    } else {
        thread_ = std::thread(&CameraCapture::CaptureThread, this);
    }
    return true;
}

void CameraCapture::Stop() {
    if (!running_.load()) return;
    running_.store(false);

    CloseBasler();
    CloseHikvision();

    if (thread_.joinable()) {
        thread_.join();
    }
}

void CameraCapture::AsyncCopyFrame(Frame* frame, const uint8_t* src, size_t size) {
    if (frame->host_pinned) {
        memcpy(frame->host_pinned, src, size);
        AsyncCopyH2D(frame->gpu_data, frame->host_pinned, size, h2d_stream_, frame->h2d_done);
    } else {
        cudaMemcpyAsync(frame->gpu_data, src, size, cudaMemcpyHostToDevice, h2d_stream_);
        cudaEventRecord(frame->h2d_done, h2d_stream_);
    }
}

bool CameraCapture::InitBasler() {
#ifdef USE_BASLER
    if (config_.camera_type != "basler") return false;
    try {
        Pylon::PylonInitialize();
        Pylon::CInstantCamera* cam = new Pylon::CInstantCamera(
            Pylon::CTlFactory::GetInstance().CreateFirstDevice());
        cam->Open();

        GenApi::CIntegerPtr width_node = cam->GetNodeMap().GetNode("Width");
        GenApi::CIntegerPtr height_node = cam->GetNodeMap().GetNode("Height");
        if (width_node) width_node->SetValue(config_.image_width);
        if (height_node) height_node->SetValue(config_.image_height);

        GenApi::CFloatPtr exposure = cam->GetNodeMap().GetNode("ExposureTime");
        if (exposure) exposure->SetValue(config_.exposure_us);

        GenApi::CEnumerationPtr trigger = cam->GetNodeMap().GetNode("TriggerMode");
        if (trigger) trigger->FromString("On");

        cam->MaxNumBuffer.SetValue(config_.buffer_count);
        cam->StartGrabbing(Pylon::GrabStrategy_LatestImageOnly);

        basler_camera_ = static_cast<void*>(cam);
        std::cout << "[CameraCapture] Basler camera opened: "
                  << cam->GetDeviceInfo().GetModelName() << "\n";
        return true;
    } catch (const GenICam::GenericException& e) {
        std::cerr << "[CameraCapture] Basler exception: " << e.GetDescription() << "\n";
        return false;
    }
#else
    return false;
#endif
}

void CameraCapture::CloseBasler() {
#ifdef USE_BASLER
    if (!basler_camera_) return;
    auto* cam = static_cast<Pylon::CInstantCamera*>(basler_camera_);
    if (cam) {
        cam->StopGrabbing();
        cam->Close();
        delete cam;
        basler_camera_ = nullptr;
    }
    Pylon::PylonTerminate();
#endif
}

bool CameraCapture::InitHikvision() {
#ifdef USE_HIKVISION
    if (config_.camera_type != "hikvision") return false;
    void* handle = nullptr;
    MV_CC_DEVICE_INFO_LIST dev_list;
    memset(&dev_list, 0, sizeof(dev_list));

    int ret = MV_CC_EnumDevices(MV_GIGE_DEVICE, &dev_list);
    if (ret != MV_OK || dev_list.nDeviceNum == 0) {
        std::cerr << "[CameraCapture] No Hikvision GigE device found\n";
        return false;
    }

    ret = MV_CC_CreateHandle(&handle, dev_list.pDeviceInfo[0]);
    if (ret != MV_OK) return false;

    ret = MV_CC_OpenDevice(handle);
    if (ret != MV_OK) {
        MV_CC_DestroyHandle(handle);
        return false;
    }

    if (config_.trigger_mode == 1) {
        MV_CC_SetEnumValue(handle, "TriggerMode", MV_TRIGGER_MODE_ON);
        MV_CC_SetEnumValue(handle, "TriggerSource", MV_TRIGGER_SOURCE_LINE0);
    }

    MV_CC_SetIntValue(handle, "Width", config_.image_width);
    MV_CC_SetIntValue(handle, "Height", config_.image_height);
    MV_CC_SetFloatValue(handle, "ExposureTime", config_.exposure_us);

    MV_CC_SetImageNodeNum(handle, config_.buffer_count);
    ret = MV_CC_StartGrabbing(handle);
    if (ret != MV_OK) {
        MV_CC_CloseDevice(handle);
        MV_CC_DestroyHandle(handle);
        return false;
    }

    hikvision_handle_ = handle;
    std::cout << "[CameraCapture] Hikvision camera opened\n";
    return true;
#else
    return false;
#endif
}

void CameraCapture::CloseHikvision() {
#ifdef USE_HIKVISION
    if (!hikvision_handle_) return;
    MV_CC_StopGrabbing(hikvision_handle_);
    MV_CC_CloseDevice(hikvision_handle_);
    MV_CC_DestroyHandle(hikvision_handle_);
    hikvision_handle_ = nullptr;
#endif
}

void CameraCapture::CaptureThread() {
#ifdef USE_BASLER
    if (basler_camera_) {
        auto* cam = static_cast<Pylon::CInstantCamera*>(basler_camera_);
        Pylon::CGrabResultPtr grab_result;
        uint64_t frame_id = 0;

        while (running_.load() && cam->IsGrabbing()) {
            if (!cam->RetrieveResult(5000, grab_result, Pylon::TimeoutHandling_ThrowException))
                continue;
            if (!grab_result->GrabSucceeded()) continue;

            Frame* frame = ring_->AcquireWriteSlot();
            const uint8_t* host_ptr = static_cast<const uint8_t*>(grab_result->GetBuffer());
            size_t sz = grab_result->GetWidth() * grab_result->GetHeight();

            AsyncCopyFrame(frame, host_ptr, sz);

            frame->meta.frame_id = frame_id++;
            frame->meta.timestamp_ns = std::chrono::high_resolution_clock::now()
                .time_since_epoch().count();
            frame->meta.width = static_cast<int>(grab_result->GetWidth());
            frame->meta.height = static_cast<int>(grab_result->GetHeight());
            frame->meta.valid = true;

            ring_->ReleaseWriteSlot(frame);
        }
        return;
    }
#endif

#ifdef USE_HIKVISION
    if (hikvision_handle_) {
        void* handle = hikvision_handle_;
        uint64_t frame_id = 0;
        MV_FRAME_OUT_INFO_EX frame_info = {};
        int payload = config_.image_width * config_.image_height;
        std::vector<uint8_t> host_buf(payload);

        while (running_.load()) {
            memset(&frame_info, 0, sizeof(frame_info));
            int ret = MV_CC_GetOneFrameTimeout(handle,
                host_buf.data(), payload, &frame_info, 5000);
            if (ret != MV_OK) continue;

            Frame* frame = ring_->AcquireWriteSlot();
            AsyncCopyFrame(frame, host_buf.data(), payload);

            frame->meta.frame_id = frame_id++;
            frame->meta.timestamp_ns = std::chrono::high_resolution_clock::now()
                .time_since_epoch().count();
            frame->meta.width = frame_info.width;
            frame->meta.height = frame_info.height;
            frame->meta.valid = true;

            ring_->ReleaseWriteSlot(frame);
        }
        return;
    }
#endif

    SimulateThread();
}

void CameraCapture::SimulateThread() {
    std::cout << "[CameraCapture] Running in SIMULATOR mode\n";
    uint64_t frame_id = 0;
    int img_size = config_.image_width * config_.image_height;

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> defect_dist(0, 99);
    std::uniform_int_distribution<uint8_t> pixel_dist(0, 255);

    while (running_.load()) {
        Frame* frame = ring_->AcquireWriteSlot();

        uint8_t* write_buf = frame->host_pinned;
        if (!write_buf) {
            ring_->ReleaseWriteSlot(frame);
            continue;
        }

        if (defect_dist(rng) < 10) {
            int x = rng() % (config_.image_width - 200);
            int y = rng() % (config_.image_height - 80);
            int w = 50 + rng() % 150;
            int h = 20 + rng() % 60;
            for (int row = y; row < y + h && row < config_.image_height; ++row) {
                for (int col = x; col < x + w && col < config_.image_width; ++col) {
                    write_buf[row * config_.image_width + col] = pixel_dist(rng);
                }
            }
        } else {
            memset(write_buf, 128, img_size);
        }

        AsyncCopyH2D(frame->gpu_data, frame->host_pinned, img_size,
                     h2d_stream_, frame->h2d_done);

        frame->meta.frame_id = frame_id++;
        frame->meta.timestamp_ns = std::chrono::high_resolution_clock::now()
            .time_since_epoch().count();
        frame->meta.width = config_.image_width;
        frame->meta.height = config_.image_height;
        frame->meta.valid = true;

        ring_->ReleaseWriteSlot(frame);

        std::this_thread::sleep_for(std::chrono::microseconds(
            static_cast<int64_t>(1000000.0 / config_.line_rate_hz)));
    }
}

}
