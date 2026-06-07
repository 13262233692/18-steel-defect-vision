#include "camera_capture.h"

#include <cuda_runtime.h>
#include <iostream>

namespace steel {

cudaError_t AllocPinnedFrameBuffer(int width, int height, uint8_t** gpu_ptr) {
    size_t bytes = static_cast<size_t>(width) * height;
    cudaError_t err = cudaMalloc(gpu_ptr, bytes);
    if (err != cudaSuccess) {
        std::cerr << "[CUDA] cudaMalloc failed: " << cudaGetErrorString(err) << "\n";
        return err;
    }
    return cudaSuccess;
}

cudaError_t FreePinnedFrameBuffer(uint8_t* gpu_ptr) {
    if (gpu_ptr) {
        cudaError_t err = cudaFree(gpu_ptr);
        if (err != cudaSuccess) {
            std::cerr << "[CUDA] cudaFree failed: " << cudaGetErrorString(err) << "\n";
        }
        return err;
    }
    return cudaSuccess;
}

cudaError_t CopyHostToGpuPinned(uint8_t* dst_gpu, const uint8_t* src_host, size_t size) {
    cudaError_t err = cudaMemcpyAsync(dst_gpu, src_host, size, cudaMemcpyHostToDevice, 0);
    if (err != cudaSuccess) {
        std::cerr << "[CUDA] cudaMemcpyAsync H2D failed: " << cudaGetErrorString(err) << "\n";
        return err;
    }
    err = cudaStreamSynchronize(0);
    return err;
}

}
