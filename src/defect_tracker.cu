#include "defect_tracker.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>

namespace steel {

__global__ void KernComputeGradientAndEigen(
    const uint8_t* __restrict__ img, int width, int height, int stride,
    float* __restrict__ eig_buf)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int y = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (x >= width - 1 || y >= height - 1) return;

    int idx = y * stride + x;
    float ix = static_cast<float>(img[idx + 1]) - static_cast<float>(img[idx - 1]);
    float iy = static_cast<float>(img[(y + 1) * stride + x]) - static_cast<float>(img[(y - 1) * stride + x]);

    float ix2 = ix * ix * 0.25f;
    float ixy = ix * iy * 0.25f;
    float iy2 = iy * iy * 0.25f;

    float a = 0.f, b = 0.f, c = 0.f;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int ny = y + dy, nx = x + dx;
            if (ny < 0 || ny >= height || nx < 0 || nx >= width) continue;
            int nidx = ny * stride + nx;
            float nix = static_cast<float>(img[nidx + 1]) - static_cast<float>(img[nidx - 1]);
            float niy = static_cast<float>(img[(ny + 1) * stride + nx]) - static_cast<float>(img[(ny - 1) * stride + nx]);
            a += nix * nix * 0.25f;
            b += nix * niy * 0.25f;
            c += niy * niy * 0.25f;
        }
    }

    float trace = a + c;
    float det = a * c - b * b;
    float disc = fmaxf(trace * trace - 4.f * det, 0.f);
    float eig = 0.5f * (trace - sqrtf(disc));

    eig_buf[y * width + x] = fmaxf(eig, 0.f);
}

__global__ void KernNonMaxSuppression(
    const float* __restrict__ eig_buf, int width, int height,
    float quality_level, float max_eig, int min_dist,
    float2* __restrict__ corners, int* __restrict__ out_count, int max_corners)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float val = eig_buf[y * width + x];
    float threshold = quality_level * max_eig;
    if (val < threshold) return;

    bool is_max = true;
    int r = min_dist / 2;
    for (int dy = -r; dy <= r && is_max; ++dy) {
        for (int dx = -r; dx <= r && is_max; ++dx) {
            if (dx == 0 && dy == 0) continue;
            int ny = y + dy, nx = x + dx;
            if (ny < 0 || ny >= height || nx < 0 || nx >= width) continue;
            if (eig_buf[ny * width + nx] > val) is_max = false;
        }
    }

    if (!is_max) return;

    int idx = atomicAdd(out_count, 1);
    if (idx < max_corners) {
        corners[idx] = make_float2(static_cast<float>(x), static_cast<float>(y));
    }
}

void CudaShiTomasiCorners(
    const uint8_t* gpu_img, int width, int height, int stride,
    float quality_level, float min_distance, int max_corners,
    float2* out_corners, int* out_count,
    cudaStream_t stream)
{
    float* d_eig = nullptr;
    cudaMalloc(&d_eig, width * height * sizeof(float));

    float* d_max_eig = nullptr;
    cudaMalloc(&d_max_eig, sizeof(float));

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    KernComputeGradientAndEigen<<<grid, block, 0, stream>>>(
        gpu_img, width, height, stride, d_eig);

    cudaMemsetAsync(d_max_eig, 0, sizeof(float), stream);
    cudaMemsetAsync(out_count, 0, sizeof(int), stream);

    KernNonMaxSuppression<<<grid, block, 0, stream>>>(
        d_eig, width, height, quality_level, 1e10f,
        static_cast<int>(min_distance),
        out_corners, out_count, max_corners);

    cudaFree(d_eig);
    cudaFree(d_max_eig);
}

__device__ float BiLerp(const uint8_t* img, float x, float y, int stride) {
    int x0 = __float2int_rn(x - 0.5f);
    int y0 = __float2int_rn(y - 0.5f);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    x0 = max(0, min(x0, stride - 1));
    x1 = max(0, min(x1, stride - 1));
    y0 = max(0, y0);
    y1 = max(0, y1);

    float fx = x - (x0 + 0.5f);
    float fy = y - (y0 + 0.5f);
    fx = fmaxf(0.f, fminf(fx, 1.f));
    fy = fmaxf(0.f, fminf(fy, 1.f));

    float v00 = static_cast<float>(img[y0 * stride + x0]);
    float v10 = static_cast<float>(img[y0 * stride + x1]);
    float v01 = static_cast<float>(img[y1 * stride + x0]);
    float v11 = static_cast<float>(img[y1 * stride + x1]);

    return v00 * (1.f - fx) * (1.f - fy) +
           v10 * fx * (1.f - fy) +
           v01 * (1.f - fx) * fy +
           v11 * fx * fy;
}

__global__ void KernLKTrack(
    const uint8_t* __restrict__ prev_img,
    const uint8_t* __restrict__ curr_img,
    int width, int height, int stride,
    const float2* __restrict__ prev_pts,
    float2* __restrict__ next_pts,
    uint8_t* __restrict__ status,
    int num_pts, int win_size, int max_iter, float epsilon)
{
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pt_idx >= num_pts) return;

    float px = prev_pts[pt_idx].x;
    float py = prev_pts[pt_idx].y;

    if (px < win_size || px >= width - win_size ||
        py < win_size || py >= height - win_size) {
        status[pt_idx] = 0;
        next_pts[pt_idx] = prev_pts[pt_idx];
        return;
    }

    float nx = px, ny = py;
    int half_w = win_size / 2;

    for (int iter = 0; iter < max_iter; ++iter) {
        float gxx = 0.f, gxy = 0.f, gyy = 0.f;
        float b1 = 0.f, b2 = 0.f;

        for (int dy = -half_w; dy <= half_w; ++dy) {
            for (int dx = -half_w; dx <= half_w; ++dx) {
                float sx = px + dx, sy = py + dy;
                float dI_dx = (BiLerp(prev_img, sx + 1.f, sy, stride) -
                               BiLerp(prev_img, sx - 1.f, sy, stride)) * 0.5f;
                float dI_dy = (BiLerp(prev_img, sx, sy + 1.f, stride) -
                               BiLerp(prev_img, sx, sy - 1.f, stride)) * 0.5f;

                float dt_val = BiLerp(curr_img, nx + dx, ny + dy, stride) -
                               BiLerp(prev_img, sx, sy, stride);

                gxx += dI_dx * dI_dx;
                gxy += dI_dx * dI_dy;
                gyy += dI_dy * dI_dy;
                b1 += -dI_dx * dt_val;
                b2 += -dI_dy * dt_val;
            }
        }

        float det = gxx * gyy - gxy * gxy;
        if (fabsf(det) < 1e-7f) {
            status[pt_idx] = 0;
            next_pts[pt_idx] = make_float2(px, py);
            return;
        }

        float inv_det = 1.f / det;
        float ddx = (gyy * b1 - gxy * b2) * inv_det;
        float ddy = (gxx * b2 - gxy * b1) * inv_det;

        nx += ddx;
        ny += ddy;

        if (nx < half_w || nx >= width - half_w ||
            ny < half_w || ny >= height - half_w) {
            status[pt_idx] = 0;
            next_pts[pt_idx] = make_float2(px, py);
            return;
        }

        if (ddx * ddx + ddy * ddy < epsilon) break;
    }

    next_pts[pt_idx] = make_float2(nx, ny);
    status[pt_idx] = 1;
}

void CudaLucasKanadeSparse(
    const uint8_t* gpu_prev, const uint8_t* gpu_curr,
    int width, int height, int stride,
    const float2* prev_pts, int num_pts,
    float2* next_pts, uint8_t* status,
    int win_size, int max_level, int max_iter, float epsilon,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (num_pts + threads - 1) / threads;

    KernLKTrack<<<blocks, threads, 0, stream>>>(
        gpu_prev, gpu_curr,
        width, height, stride,
        prev_pts, next_pts, status,
        num_pts, win_size, max_iter, epsilon);
}

}
