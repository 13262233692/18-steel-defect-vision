#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <string>
#include <vector>

namespace steel {

struct Detection;

struct FlowVector {
    float dx = 0.f;
    float dy = 0.f;
    bool valid = false;
};

struct TrackedDefect {
    int track_id = -1;
    int class_id = -1;
    float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
    float total_length = 0.f;
    int frame_count = 1;
    uint64_t first_frame_id = 0;
    uint64_t last_frame_id = 0;
    float confidence = 0.f;
    bool active = true;
};

struct TrackerConfig {
    int overlap_rows = 128;
    int max_corners = 64;
    float corner_quality = 0.01f;
    float corner_min_distance = 10.f;
    int lk_win_size = 21;
    int lk_max_level = 3;
    float lk_epsilon = 0.03f;
    int lk_max_iter = 30;
    float flow_dy_threshold = 5.f;
    float iou_threshold = 0.15f;
    float vertical_alignment_tol = 30.f;
    int max_missing_frames = 3;
};

struct OverlapPatch {
    uint8_t* gpu_data = nullptr;
    int width = 0;
    int height = 0;
    int stride = 0;
};

void CudaShiTomasiCorners(
    const uint8_t* gpu_img, int width, int height, int stride,
    float quality_level, float min_distance, int max_corners,
    float2* out_corners, int* out_count,
    cudaStream_t stream);

void CudaLucasKanadeSparse(
    const uint8_t* gpu_prev, const uint8_t* gpu_curr,
    int width, int height, int stride,
    const float2* prev_pts, int num_pts,
    float2* next_pts, uint8_t* status,
    int win_size, int max_level, int max_iter, float epsilon,
    cudaStream_t stream);

class DefectTracker {
public:
    explicit DefectTracker(const TrackerConfig& cfg);
    ~DefectTracker();

    void Update(const std::vector<Detection>& dets,
                const OverlapPatch& prev_overlap,
                const OverlapPatch& curr_overlap,
                uint64_t frame_id,
                cudaStream_t stream);

    const std::vector<TrackedDefect>& ActiveTracks() const { return tracks_; }
    std::vector<TrackedDefect> FinalizeTracks();

    void PrintStats() const;

private:
    int AssignTrackId();
    void AssociateDetections(const std::vector<Detection>& dets, uint64_t frame_id);
    std::vector<FlowVector> ComputeFlowInRoi(
        const OverlapPatch& prev, const OverlapPatch& curr,
        float rx1, float ry1, float rx2, float ry2,
        cudaStream_t stream);

    TrackerConfig config_;
    std::vector<TrackedDefect> tracks_;
    int next_track_id_ = 0;
};

}
