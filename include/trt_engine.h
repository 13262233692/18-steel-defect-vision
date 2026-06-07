#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>

#include "config.h"

namespace nvinfer1 {
class IRuntime;
class ICudaEngine;
class IExecutionContext;
}

namespace steel {

class Logger;

class TrtEngine {
public:
    explicit TrtEngine(const TritConfig& cfg);
    ~TrtEngine();

    bool BuildOrLoadEngine();
    bool Infer(cudaStream_t stream);

    const float* OutputGpuPtr() const { return output_gpu_; }
    float* InputGpuPtr() const { return input_gpu_; }
    int OutputSize() const { return output_size_; }
    int OutputShape(int dim) const;

private:
    bool BuildEngineFromOnnx();
    bool SerializeEngine(const std::string& path);
    bool DeserializeEngine(const std::string& path);

    TritConfig config_;
    std::unique_ptr<Logger> logger_;

    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;

    void* bindings_[2] = {nullptr, nullptr};
    float* input_gpu_ = nullptr;
    float* output_gpu_ = nullptr;
    int input_size_ = 0;
    int output_size_ = 0;
    int input_index_ = -1;
    int output_index_ = -1;
};

}
