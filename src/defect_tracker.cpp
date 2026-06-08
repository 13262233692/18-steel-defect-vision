#include "defect_tracker.h"
#include "yolo_postprocess.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>

namespace steel {

DefectTracker::DefectTracker(const TrackerConfig& cfg)
    : config_(cfg) {}

DefectTracker::~DefectTracker() = default;

int DefectTracker::AssignTrackId() {
    return next_track_id_++;
}

static float ComputeIoU(const TrackedDefect& t, const Detection& d) {
    float ix1 = std::max(t.x1, d.x1);
    float iy1 = std::max(t.y1, d.y1);
    float ix2 = std::min(t.x2, d.x2);
    float iy2 = std::min(t.y2, d.y2);

    float iw = std::max(0.f, ix2 - ix1);
    float ih = std::max(0.f, iy2 - iy1);
    float inter = iw * ih;

    float a1 = (t.x2 - t.x1) * (t.y2 - t.y1);
    float a2 = (d.x2 - d.x1) * (d.y2 - d.y1);

    return inter / (a1 + a2 - inter + 1e-6f);
}

static bool VerticallyAligned(const TrackedDefect& t, const Detection& d, float tol) {
    float tcx = (t.x1 + t.x2) * 0.5f;
    float dcx = (d.x1 + d.x2) * 0.5f;
    return fabsf(tcx - dcx) < tol;
}

std::vector<FlowVector> DefectTracker::ComputeFlowInRoi(
    const OverlapPatch& prev, const OverlapPatch& curr,
    float rx1, float ry1, float rx2, float ry2,
    cudaStream_t stream)
{
    std::vector<FlowVector> result;

    if (!prev.gpu_data || !curr.gpu_data || prev.width <= 0 || prev.height <= 0)
        return result;

    float2* d_corners = nullptr;
    int* d_count = nullptr;
    cudaMalloc(&d_corners, config_.max_corners * sizeof(float2));
    cudaMalloc(&d_count, sizeof(int));

    CudaShiTomasiCorners(
        prev.gpu_data, prev.width, prev.height, prev.stride,
        config_.corner_quality, config_.corner_min_distance,
        config_.max_corners, d_corners, d_count, stream);

    int h_count = 0;
    cudaMemcpyAsync(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    if (h_count <= 0) {
        cudaFree(d_corners);
        cudaFree(d_count);
        return result;
    }

    h_count = std::min(h_count, config_.max_corners);

    float2* d_next = nullptr;
    uint8_t* d_status = nullptr;
    cudaMalloc(&d_next, h_count * sizeof(float2));
    cudaMalloc(&d_status, h_count * sizeof(uint8_t));

    CudaLucasKanadeSparse(
        prev.gpu_data, curr.gpu_data,
        prev.width, prev.height, prev.stride,
        d_corners, h_count, d_next, d_status,
        config_.lk_win_size, config_.lk_max_level,
        config_.lk_max_iter, config_.lk_epsilon, stream);

    std::vector<float2> h_prev(h_count), h_next(h_count);
    std::vector<uint8_t> h_status(h_count);

    cudaMemcpyAsync(h_prev.data(), d_corners, h_count * sizeof(float2),
        cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_next.data(), d_next, h_count * sizeof(float2),
        cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_status.data(), d_status, h_count * sizeof(uint8_t),
        cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    for (int i = 0; i < h_count; ++i) {
        if (!h_status[i]) continue;
        float px = h_prev[i].x, py = h_prev[i].y;
        if (px < rx1 || px > rx2 || py < ry1 || py > ry2) continue;

        FlowVector fv;
        fv.dx = h_next[i].x - h_prev[i].x;
        fv.dy = h_next[i].y - h_prev[i].y;
        fv.valid = true;
        result.push_back(fv);
    }

    cudaFree(d_corners);
    cudaFree(d_count);
    cudaFree(d_next);
    cudaFree(d_status);

    return result;
}

void DefectTracker::AssociateDetections(const std::vector<Detection>& dets, uint64_t frame_id) {
    std::vector<bool> det_matched(dets.size(), false);
    std::vector<bool> trk_matched(tracks_.size(), false);

    for (size_t ti = 0; ti < tracks_.size(); ++ti) {
        if (!tracks_[ti].active) continue;

        float best_iou = 0.f;
        int best_di = -1;

        for (size_t di = 0; di < dets.size(); ++di) {
            if (det_matched[di]) continue;
            if (tracks_[ti].class_id != dets[di].class_id) continue;

            float iou = ComputeIoU(tracks_[ti], dets[di]);
            if (iou > best_iou) {
                best_iou = iou;
                best_di = static_cast<int>(di);
            }
        }

        if (best_iou >= config_.iou_threshold && best_di >= 0) {
            const auto& d = dets[best_di];
            TrackedDefect& t = tracks_[ti];

            float old_h = t.y2 - t.y1;
            t.y2 = d.y2;
            t.x1 = std::min(t.x1, d.x1);
            t.x2 = std::max(t.x2, d.x2);
            t.confidence = std::max(t.confidence, d.confidence);

            float new_h = d.y2 - d.y1;
            float vertical_ext = new_h;
            if (d.y1 >= t.y1 && d.y2 <= t.y2) {
                vertical_ext = 0.f;
            } else {
                vertical_ext = std::max(0.f, d.y2 - t.y2);
            }
            t.total_length += vertical_ext;

            t.frame_count++;
            t.last_frame_id = frame_id;
            trk_matched[ti] = true;
            det_matched[best_di] = true;
        }
    }

    for (size_t ti = 0; ti < tracks_.size(); ++ti) {
        if (!tracks_[ti].active) continue;
        if (!trk_matched[ti]) {
            auto gap = frame_id - tracks_[ti].last_frame_id;
            if (gap > static_cast<uint64_t>(config_.max_missing_frames)) {
                tracks_[ti].active = false;
            }
        }
    }

    for (size_t di = 0; di < dets.size(); ++di) {
        if (det_matched[di]) continue;

        bool merged = false;
        for (auto& t : tracks_) {
            if (!t.active) continue;
            if (t.class_id != dets[di].class_id) continue;
            if (!VerticallyAligned(t, dets[di], config_.vertical_alignment_tol)) continue;

            float gap_top = fabsf(dets[di].y1 - t.y2);
            float gap_bot = fabsf(t.y1 - dets[di].y2);

            if (gap_top < config_.vertical_alignment_tol ||
                gap_bot < config_.vertical_alignment_tol) {
                float ext = std::max(0.f, dets[di].y2 - t.y2);
                t.x1 = std::min(t.x1, dets[di].x1);
                t.x2 = std::max(t.x2, dets[di].x2);
                t.y2 = std::max(t.y2, dets[di].y2);
                t.total_length += ext;
                t.confidence = std::max(t.confidence, dets[di].confidence);
                t.frame_count++;
                t.last_frame_id = frame_id;
                merged = true;
                break;
            }
        }

        if (!merged) {
            TrackedDefect nt;
            nt.track_id = AssignTrackId();
            nt.class_id = dets[di].class_id;
            nt.x1 = dets[di].x1;
            nt.y1 = dets[di].y1;
            nt.x2 = dets[di].x2;
            nt.y2 = dets[di].y2;
            nt.total_length = dets[di].y2 - dets[di].y1;
            nt.frame_count = 1;
            nt.first_frame_id = frame_id;
            nt.last_frame_id = frame_id;
            nt.confidence = dets[di].confidence;
            nt.active = true;
            tracks_.push_back(nt);
        }
    }
}

void DefectTracker::Update(const std::vector<Detection>& dets,
                            const OverlapPatch& prev_overlap,
                            const OverlapPatch& curr_overlap,
                            uint64_t frame_id,
                            cudaStream_t stream)
{
    if (!dets.empty() && prev_overlap.gpu_data && curr_overlap.gpu_data) {
        for (const auto& d : dets) {
            if (d.y1 >= prev_overlap.height) continue;

            auto flows = ComputeFlowInRoi(
                prev_overlap, curr_overlap,
                d.x1, d.y1, d.x2, d.y2,
                stream);

            (void)flows;
        }
    }

    AssociateDetections(dets, frame_id);
}

std::vector<TrackedDefect> DefectTracker::FinalizeTracks() {
    for (auto& t : tracks_) {
        t.active = false;
    }
    return tracks_;
}

void DefectTracker::PrintStats() const {
    int active = 0, total = static_cast<int>(tracks_.size());
    for (const auto& t : tracks_) {
        if (t.active) active++;
    }

    std::cout << "[DefectTracker] Tracks: " << total
              << " (active=" << active << ")\n";

    if (total == 0) return;

    std::cout << "  ID | Class | Length(px) | Frames | Conf\n";
    std::cout << "  ---+-------+------------+--------+------\n";
    for (const auto& t : tracks_) {
        std::cout << "  " << std::setw(3) << t.track_id
                  << " | " << std::setw(5) << t.class_id
                  << " | " << std::setw(10) << std::fixed << std::setprecision(1) << t.total_length
                  << " | " << std::setw(6) << t.frame_count
                  << " | " << std::setw(4) << std::setprecision(2) << t.confidence
                  << "\n";
    }
}

}
