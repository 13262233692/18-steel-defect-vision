#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include "config.h"

namespace steel {

struct FrameMeta {
    uint64_t frame_id = 0;
    uint64_t timestamp_ns = 0;
    int width = 0;
    int height = 0;
    bool valid = false;
};

struct Frame {
    uint8_t* gpu_data = nullptr;
    FrameMeta meta;
};

class RingBuffer {
public:
    explicit RingBuffer(int capacity, int width, int height);
    ~RingBuffer();

    Frame* AcquireWriteSlot();
    void ReleaseWriteSlot(Frame* frame);
    Frame* AcquireReadSlot();
    void ReleaseReadSlot(Frame* frame);

    int Capacity() const { return capacity_; }

private:
    int capacity_;
    int slot_size_;
    std::vector<Frame> frames_;
    std::vector<std::atomic<int>> slot_state_;
    int write_idx_ = 0;
    int read_idx_ = 0;
    std::mutex write_mtx_;
    std::mutex read_mtx_;
    std::condition_variable write_cv_;
    std::condition_variable read_cv_;
};

class CameraCapture {
public:
    using Callback = std::function<void(Frame*)>;

    explicit CameraCapture(const CameraConfig& cfg, RingBuffer* ring);
    ~CameraCapture();

    bool Start();
    void Stop();
    bool IsRunning() const { return running_.load(); }

private:
    void CaptureThread();
    bool InitBasler();
    bool InitHikvision();
    void CloseBasler();
    void CloseHikvision();
    void SimulateThread();

    CameraConfig config_;
    RingBuffer* ring_;
    std::atomic<bool> running_{false};
    std::thread thread_;

    void* basler_camera_ = nullptr;
    void* hikvision_handle_ = nullptr;
};

cudaError_t AllocPinnedFrameBuffer(int width, int height, uint8_t** gpu_ptr);
cudaError_t FreePinnedFrameBuffer(uint8_t* gpu_ptr);
cudaError_t CopyHostToGpuPinned(uint8_t* dst_gpu, const uint8_t* src_host, size_t size);

}
