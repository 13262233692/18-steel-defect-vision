#include "yolo_postprocess.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>

namespace steel {

__global__ void KernResizeBilinear(
    const uint8_t* __restrict__ src, int src_w, int src_h,
    float* __restrict__ dst, int dst_w, int dst_h,
    int pad_w, int pad_h, float scale_x, float scale_y)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dst_w || dy >= dst_h) return;

    if (dx < pad_w || dx >= dst_w - pad_w ||
        dy < pad_h || dy >= dst_h - pad_h) {
        int c3 = dy * dst_w + dx;
        dst[c3] = 0.f;
        dst[dst_w * dst_h + c3] = 0.f;
        dst[2 * dst_w * dst_h + c3] = 0.f;
        return;
    }

    float sx = (dx - pad_w) * scale_x;
    float sy = (dy - pad_h) * scale_y;

    int x0 = __float2int_rn(sx);
    int y0 = __float2int_rn(sy);
    int x1 = min(x0 + 1, src_w - 1);
    int y1 = min(y0 + 1, src_h - 1);
    x0 = max(0, x0);
    y0 = max(0, y0);

    float fx = sx - x0;
    float fy = sy - y0;
    float w00 = (1.f - fx) * (1.f - fy);
    float w10 = fx * (1.f - fy);
    float w01 = (1.f - fx) * fy;
    float w11 = fx * fy;

    int idx00 = y0 * src_w + x0;
    int idx10 = y0 * src_w + x1;
    int idx01 = y1 * src_w + x0;
    int idx11 = y1 * src_w + x1;

    int c3 = dy * dst_w + dx;
    float gray = static_cast<float>(src[idx00]) * w00 +
                 static_cast<float>(src[idx10]) * w10 +
                 static_cast<float>(src[idx01]) * w01 +
                 static_cast<float>(src[idx11]) * w11;

    dst[c3] = gray / 255.f;
    dst[dst_w * dst_h + c3] = gray / 255.f;
    dst[2 * dst_w * dst_h + c3] = gray / 255.f;
}

void CudaPreprocessResize(
    const uint8_t* src_gpu, int src_w, int src_h,
    float* dst_gpu, int dst_w, int dst_h,
    cudaStream_t stream)
{
    float scale_x = static_cast<float>(src_w) / (dst_w);
    float scale_y = static_cast<float>(src_h) / (dst_h);

    float ratio = fminf(static_cast<float>(dst_w) / src_w,
                        static_cast<float>(dst_h) / src_h);
    int new_w = static_cast<int>(src_w * ratio);
    int new_h = static_cast<int>(src_h * ratio);
    int pad_w = (dst_w - new_w) / 2;
    int pad_h = (dst_h - new_h) / 2;

    float sx = static_cast<float>(src_w) / new_w;
    float sy = static_cast<float>(src_h) / new_h;

    dim3 block(16, 16);
    dim3 grid((dst_w + 15) / 16, (dst_h + 15) / 16);

    KernResizeBilinear<<<grid, block, 0, stream>>>(
        src_gpu, src_w, src_h, dst_gpu, dst_w, dst_h,
        pad_w, pad_h, sx, sy);
}

__global__ void KernDecodeYoloV8(
    const float* __restrict__ output,
    int num_rows, int num_classes,
    float score_thresh,
    float* __restrict__ scores,
    int* __restrict__ class_ids,
    float* __restrict__ bboxes)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= num_rows) return;

    int offset = r * (4 + num_classes);

    float cx = output[offset + 0];
    float cy = output[offset + 1];
    float w = output[offset + 2];
    float h = output[offset + 3];

    int best_cls = 0;
    float best_score = output[offset + 4];
    for (int c = 1; c < num_classes; ++c) {
        float s = output[offset + 4 + c];
        if (s > best_score) {
            best_score = s;
            best_cls = c;
        }
    }

    scores[r] = best_score;
    class_ids[r] = best_cls;
    bboxes[r * 4 + 0] = cx - w * 0.5f;
    bboxes[r * 4 + 1] = cy - h * 0.5f;
    bboxes[r * 4 + 2] = cx + w * 0.5f;
    bboxes[r * 4 + 3] = cy + h * 0.5f;
}

std::vector<Detection> DecodeYoloV8Output(
    const float* gpu_output,
    int num_rows,
    int num_classes,
    float score_thresh,
    float nms_thresh,
    const PreprocessParams& params,
    cudaStream_t stream)
{
    float* d_scores = nullptr;
    int* d_cls = nullptr;
    float* d_bbox = nullptr;

    cudaMalloc(&d_scores, num_rows * sizeof(float));
    cudaMalloc(&d_cls, num_rows * sizeof(int));
    cudaMalloc(&d_bbox, num_rows * 4 * sizeof(float));

    int threads = 256;
    int blocks = (num_rows + threads - 1) / threads;
    KernDecodeYoloV8<<<blocks, threads, 0, stream>>>(
        gpu_output, num_rows, num_classes, score_thresh,
        d_scores, d_cls, d_bbox);

    std::vector<float> h_scores(num_rows);
    std::vector<int> h_cls(num_rows);
    std::vector<float> h_bbox(num_rows * 4);

    cudaMemcpyAsync(h_scores.data(), d_scores, num_rows * sizeof(float),
        cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_cls.data(), d_cls, num_rows * sizeof(int),
        cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_bbox.data(), d_bbox, num_rows * 4 * sizeof(float),
        cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaFree(d_scores);
    cudaFree(d_cls);
    cudaFree(d_bbox);

    std::vector<Detection> candidates;
    candidates.reserve(num_rows);
    for (int i = 0; i < num_rows; ++i) {
        if (h_scores[i] < score_thresh) continue;
        Detection d;
        d.x1 = h_bbox[i * 4 + 0];
        d.y1 = h_bbox[i * 4 + 1];
        d.x2 = h_bbox[i * 4 + 2];
        d.y2 = h_bbox[i * 4 + 3];
        d.confidence = h_scores[i];
        d.class_id = h_cls[i];
        candidates.push_back(d);
    }

    std::sort(candidates.begin(), candidates.end(),
        [](const Detection& a, const Detection& b) {
            return a.confidence > b.confidence;
        });

    std::vector<Detection> result;
    std::vector<bool> suppressed(candidates.size(), false);

    for (size_t i = 0; i < candidates.size(); ++i) {
        if (suppressed[i]) continue;
        result.push_back(candidates[i]);

        for (size_t j = i + 1; j < candidates.size(); ++j) {
            if (suppressed[j]) continue;
            if (candidates[i].class_id != candidates[j].class_id) continue;

            float ix1 = std::max(candidates[i].x1, candidates[j].x1);
            float iy1 = std::max(candidates[i].y1, candidates[j].y1);
            float ix2 = std::min(candidates[i].x2, candidates[j].x2);
            float iy2 = std::min(candidates[i].y2, candidates[j].y2);

            float iw = std::max(0.f, ix2 - ix1);
            float ih = std::max(0.f, iy2 - iy1);
            float inter = iw * ih;

            float a1 = (candidates[i].x2 - candidates[i].x1) *
                        (candidates[i].y2 - candidates[i].y1);
            float a2 = (candidates[j].x2 - candidates[j].x1) *
                        (candidates[j].y2 - candidates[j].y1);
            float iou = inter / (a1 + a2 - inter + 1e-6f);

            if (iou > nms_thresh) {
                suppressed[j] = true;
            }
        }
    }

    float ratio = std::min(static_cast<float>(params.orig_w) / (params.orig_w - 2 * params.pad_w + 1e-6f),
                           static_cast<float>(params.orig_h) / (params.orig_h - 2 * params.pad_h + 1e-6f));

    for (auto& det : result) {
        det.x1 = (det.x1 - params.pad_w) / ratio;
        det.y1 = (det.y1 - params.pad_h) / ratio;
        det.x2 = (det.x2 - params.pad_w) / ratio;
        det.y2 = (det.y2 - params.pad_h) / ratio;

        det.x1 = std::max(0.f, std::min(det.x1, static_cast<float>(params.orig_w)));
        det.y1 = std::max(0.f, std::min(det.y1, static_cast<float>(params.orig_h)));
        det.x2 = std::max(0.f, std::min(det.x2, static_cast<float>(params.orig_w)));
        det.y2 = std::max(0.f, std::min(det.y2, static_cast<float>(params.orig_h)));
    }

    return result;
}

}
