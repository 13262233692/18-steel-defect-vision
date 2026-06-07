#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cuda_runtime.h>
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
    uint8_t* host_pinned = nullptr;
    cudaEvent_t h2d_done = nullptr;
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
    int Width() const { return width_; }
    int Height() const { return height_; }

private:
    int capacity_;
    int width_;
    int height_;
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
    cudaStream_t H2DStream() const { return h2d_stream_; }

private:
    void CaptureThread();
    bool InitBasler();
    bool InitHikvision();
    void CloseBasler();
    void CloseHikvision();
    void SimulateThread();
    void AsyncCopyFrame(Frame* frame, const uint8_t* src, size_t size);

    CameraConfig config_;
    RingBuffer* ring_;
    std::atomic<bool> running_{false};
    std::thread thread_;

    cudaStream_t h2d_stream_ = nullptr;

    void* basler_camera_ = nullptr;
    void* hikvision_handle_ = nullptr;
};

}
