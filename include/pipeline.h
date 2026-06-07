#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "config.h"

namespace steel {

class CameraCapture;
class RingBuffer;
class TrtEngine;

struct PipelineStats {
    uint64_t total_frames = 0;
    uint64_t defect_frames = 0;
    double avg_capture_ms = 0.0;
    double avg_preprocess_ms = 0.0;
    double avg_infer_ms = 0.0;
    double avg_postprocess_ms = 0.0;
    double avg_e2e_ms = 0.0;
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
    void OnFrameArrived(class Frame* frame);

    PipelineConfig config_;
    std::unique_ptr<RingBuffer> ring_;
    std::unique_ptr<CameraCapture> capture_;
    std::unique_ptr<TrtEngine> engine_;

    std::atomic<bool> running_{false};
    std::thread infer_thread_;

    PipelineStats stats_;
    std::chrono::time_point<std::chrono::high_resolution_clock> frame_start_;
};

}
