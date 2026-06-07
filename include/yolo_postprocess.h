#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <string>
#include <vector>

#include "config.h"

namespace cv {
class Mat;
}

namespace steel {

struct Detection {
    float x1, y1, x2, y2;
    float confidence;
    int class_id;
};

struct PreprocessParams {
    float scale_x;
    float scale_y;
    int pad_w;
    int pad_h;
    int orig_w;
    int orig_h;
};

void CudaPreprocessResize(
    const uint8_t* src_gpu, int src_w, int src_h,
    float* dst_gpu, int dst_w, int dst_h,
    cudaStream_t stream);

std::vector<Detection> DecodeYoloV8Output(
    const float* gpu_output,
    int num_rows,
    int num_classes,
    float score_thresh,
    float nms_thresh,
    const PreprocessParams& params,
    cudaStream_t stream);

void DrawDetections(
    cv::Mat& image,
    const std::vector<Detection>& dets,
    const std::vector<std::string>& class_names);

std::vector<std::string> DefaultDefectClassNames();

}
