#pragma once

#include <atomic>
#include <chrono>
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>

#include "config.h"

namespace steel {

class CameraCapture;
class RingBuffer;
class TrtEngine;
class DefectTracker;
struct OverlapPatch;

struct PipelineStats {
    uint64_t total_frames = 0;
    uint64_t defect_frames = 0;
    double avg_capture_ms = 0.0;
    double avg_preprocess_ms = 0.0;
    double avg_infer_ms = 0.0;
    double avg_postprocess_ms = 0.0;
    double avg_e2e_ms = 0.0;
    int total_unique_defects = 0;
};

class Pipeline {
public:
    explicit Pipeline(const PipelineConfig& cfg);
    ~Pipeline();

    bool Init();
    bool Start();
    void Stop();
    bool IsRunning() const { return running_.load(); }

    PipelineStats GetStats() const;

private:
    void InferenceLoop();
    void CacheOverlapRegion(const uint8_t* gpu_frame, int width, int height,
                            cudaStream_t stream);

    PipelineConfig config_;
    std::unique_ptr<RingBuffer> ring_;
    std::unique_ptr<CameraCapture> capture_;
    std::unique_ptr<TrtEngine> engine_;
    std::unique_ptr<DefectTracker> tracker_;

    cudaStream_t preprocess_stream_ = nullptr;
    cudaStream_t infer_stream_ = nullptr;
    cudaStream_t postprocess_stream_ = nullptr;
    cudaStream_t tracker_stream_ = nullptr;

    cudaEvent_t preprocess_done_ = nullptr;
    cudaEvent_t infer_done_ = nullptr;

    uint8_t* prev_overlap_gpu_ = nullptr;
    uint8_t* curr_overlap_gpu_ = nullptr;
    int overlap_width_ = 0;
    int overlap_height_ = 0;
    bool has_prev_overlap_ = false;

    std::atomic<bool> running_{false};
    std::thread infer_thread_;

    PipelineStats stats_;
};

}
