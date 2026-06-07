#include "trt_engine.h"

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>

#include <fstream>
#include <iostream>

namespace steel {

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TRT] " << msg << "\n";
        }
    }
};

TrtEngine::TrtEngine(const TritConfig& cfg)
    : config_(cfg), logger_(std::make_unique<Logger>()) {}

TrtEngine::~TrtEngine() {
    if (context_) {
        context_->destroy();
        context_ = nullptr;
    }
    if (engine_) {
        engine_->destroy();
        engine_ = nullptr;
    }
    if (runtime_) {
        runtime_->destroy();
        runtime_ = nullptr;
    }
    if (input_gpu_) cudaFree(input_gpu_);
    if (output_gpu_) cudaFree(output_gpu_);
}

bool TrtEngine::BuildOrLoadEngine() {
    cudaSetDevice(config_.gpu_id);

    std::ifstream engine_file(config_.engine_path, std::ios::binary);
    if (engine_file.good()) {
        std::cout << "[TrtEngine] Found cached engine: " << config_.engine_path << "\n";
        if (DeserializeEngine(config_.engine_path)) {
            return true;
        }
        std::cout << "[TrtEngine] Cached engine load failed, rebuilding...\n";
    }

    if (!BuildEngineFromOnnx()) {
        std::cerr << "[TrtEngine] Failed to build engine from ONNX\n";
        return false;
    }

    SerializeEngine(config_.engine_path);
    return true;
}

bool TrtEngine::BuildEngineFromOnnx() {
    auto builder = nvinfer1::createInferBuilder(*logger_);
    if (!builder) {
        std::cerr << "[TrtEngine] Failed to create builder\n";
        return false;
    }

    const uint32_t explicit_batch = 1U << static_cast<uint32_t>(
        nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = builder->createNetworkV2(explicit_batch);
    if (!network) {
        std::cerr << "[TrtEngine] Failed to create network\n";
        builder->destroy();
        return false;
    }

    auto parser = nvonnxparser::createParser(*network, *logger_);
    if (!parser) {
        std::cerr << "[TrtEngine] Failed to create ONNX parser\n";
        network->destroy();
        builder->destroy();
        return false;
    }

    if (!parser->parseFromFile(config_.onnx_path.c_str(),
            static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::cerr << "[TrtEngine] Failed to parse ONNX: " << config_.onnx_path << "\n";
        for (int i = 0; i < parser->getNbErrors(); ++i) {
            std::cerr << "  Error: " << parser->getError(i)->desc() << "\n";
        }
        parser->destroy();
        network->destroy();
        builder->destroy();
        return false;
    }
    std::cout << "[TrtEngine] ONNX parsed successfully\n";
    parser->destroy();

    auto config = builder->createBuilderConfig();
    config->setMaxWorkspaceSize(static_cast<size_t>(config_.max_workspace_mb) * 1024 * 1024);

    if (config_.fp16 && builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        std::cout << "[TrtEngine] FP16 enabled\n";
    }

    if (config_.int8 && builder->platformHasFastInt8()) {
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
        std::cout << "[TrtEngine] INT8 enabled (calibration cache: "
                  << config_.calib_cache << ")\n";
    }

    auto profile = builder->createOptimizationProfile();
    auto input_tensor = network->getInput(0);
    auto input_dims = input_tensor->getDimensions();

    nvinfer1::Dims4 min_dims(config_.batch_size, input_dims.d[1],
        config_.input_height, config_.input_width);
    nvinfer1::Dims4 opt_dims(config_.batch_size, input_dims.d[1],
        config_.input_height, config_.input_width);
    nvinfer1::Dims4 max_dims(config_.batch_size, input_dims.d[1],
        config_.input_height, config_.input_width);

    profile->setDimensions(input_tensor->getName(),
        nvinfer1::OptProfileSelector::kMIN, min_dims);
    profile->setDimensions(input_tensor->getName(),
        nvinfer1::OptProfileSelector::kOPT, opt_dims);
    profile->setDimensions(input_tensor->getName(),
        nvinfer1::OptProfileSelector::kMAX, max_dims);
    config->addOptimizationProfile(profile);

    std::cout << "[TrtEngine] Building engine (this may take a while)...\n";
    auto* plan = builder->buildSerializedNetwork(*network, *config);
    if (!plan) {
        std::cerr << "[TrtEngine] Failed to build serialized network\n";
        config->destroy();
        network->destroy();
        builder->destroy();
        return false;
    }

    runtime_ = nvinfer1::createInferRuntime(*logger_);
    engine_ = runtime_->deserializeCudaEngine(plan->getData(), plan->getSize());

    plan->destroy();
    config->destroy();
    network->destroy();
    builder->destroy();

    if (!engine_) {
        std::cerr << "[TrtEngine] Failed to deserialize built engine\n";
        return false;
    }

    context_ = engine_->createExecutionContext();
    std::cout << "[TrtEngine] Engine built and context created\n";
    return true;
}

bool TrtEngine::SerializeEngine(const std::string& path) {
    if (!engine_) return false;
    auto* serialized = engine_->serialize();
    if (!serialized) {
        std::cerr << "[TrtEngine] Failed to serialize engine\n";
        return false;
    }

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        std::cerr << "[TrtEngine] Cannot open file for writing: " << path << "\n";
        serialized->destroy();
        return false;
    }

    out.write(static_cast<const char*>(serialized->getData()), serialized->getSize());
    out.close();
    serialized->destroy();
    std::cout << "[TrtEngine] Engine serialized to: " << path << "\n";
    return true;
}

bool TrtEngine::DeserializeEngine(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;

    in.seekg(0, std::ios::end);
    size_t size = static_cast<size_t>(in.tellg());
    in.seekg(0, std::ios::beg);

    std::vector<char> buf(size);
    in.read(buf.data(), size);
    in.close();

    if (!runtime_) {
        runtime_ = nvinfer1::createInferRuntime(*logger_);
    }

    engine_ = runtime_->deserializeCudaEngine(buf.data(), size);
    if (!engine_) {
        std::cerr << "[TrtEngine] Failed to deserialize engine\n";
        return false;
    }

    context_ = engine_->createExecutionContext();
    std::cout << "[TrtEngine] Engine deserialized successfully\n";
    return true;
}

bool TrtEngine::Infer(cudaStream_t stream) {
    if (!context_) return false;

    if (!input_gpu_) {
        input_size_ = config_.batch_size * 3 * config_.input_height * config_.input_width;
        output_size_ = 0;

        int nb = engine_->getNbBindings();
        for (int i = 0; i < nb; ++i) {
            if (engine_->bindingIsInput(i)) {
                input_index_ = i;
            } else {
                output_index_ = i;
                auto dims = engine_->getBindingDimensions(i);
                output_size_ = 1;
                for (int d = 0; d < dims.nbDims; ++d) {
                    output_size_ *= dims.d[d];
                }
            }
        }

        if (output_size_ <= 0) {
            output_size_ = config_.batch_size * (4 + 20);
        }

        cudaMalloc(&input_gpu_, input_size_ * sizeof(float));
        cudaMalloc(&output_gpu_, output_size_ * sizeof(float));

        bindings_[input_index_] = input_gpu_;
        bindings_[output_index_] = output_gpu_;

        std::cout << "[TrtEngine] Alloc: input=" << input_size_
                  << " output=" << output_size_ << "\n";
    }

    context_->enqueueV2(bindings_, stream, nullptr);

    return true;
}

int TrtEngine::OutputShape(int dim) const {
    if (!engine_) return 0;
    auto dims = engine_->getBindingDimensions(output_index_);
    if (dim < dims.nbDims) return dims.d[dim];
    return 0;
}

}
