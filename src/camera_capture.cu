#include "camera_capture.h"

#include <cuda_runtime.h>
#include <iostream>

namespace steel {

cudaError_t AllocPinnedFrameBuffer(int width, int height, uint8_t** gpu_ptr, uint8_t** host_ptr, cudaEvent_t* event) {
    size_t bytes = static_cast<size_t>(width) * height;

    cudaError_t err = cudaMalloc(gpu_ptr, bytes);
    if (err != cudaSuccess) {
        std::cerr << "[CUDA] cudaMalloc failed: " << cudaGetErrorString(err) << "\n";
        return err;
    }

    err = cudaMallocHost(host_ptr, bytes);
    if (err != cudaSuccess) {
        std::cerr << "[CUDA] cudaMallocHost failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(*gpu_ptr);
        *gpu_ptr = nullptr;
        return err;
    }

    err = cudaEventCreateWithFlags(event, cudaEventDisableTiming);
    if (err != cudaSuccess) {
        std::cerr << "[CUDA] cudaEventCreate failed: " << cudaGetErrorString(err) << "\n";
        cudaFreeHost(*host_ptr);
        cudaFree(*gpu_ptr);
        *gpu_ptr = nullptr;
        *host_ptr = nullptr;
        return err;
    }

    return cudaSuccess;
}

cudaError_t FreePinnedFrameBuffer(uint8_t* gpu_ptr, uint8_t* host_ptr, cudaEvent_t event) {
    if (event) {
        cudaEventDestroy(event);
    }
    if (host_ptr) {
        cudaFreeHost(host_ptr);
    }
    if (gpu_ptr) {
        cudaFree(gpu_ptr);
    }
    return cudaSuccess;
}

cudaError_t AsyncCopyH2D(uint8_t* dst_gpu, const uint8_t* src_host, size_t size,
                          cudaStream_t stream, cudaEvent_t done_event) {
    cudaError_t err = cudaMemcpyAsync(dst_gpu, src_host, size, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        std::cerr << "[CUDA] cudaMemcpyAsync H2D failed: " << cudaGetErrorString(err) << "\n";
        return err;
    }
    err = cudaEventRecord(done_event, stream);
    if (err != cudaSuccess) {
        std::cerr << "[CUDA] cudaEventRecord failed: " << cudaGetErrorString(err) << "\n";
    }
    return err;
}

}
