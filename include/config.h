#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace steel {

struct CameraConfig {
    std::string camera_type = "basler";
    std::string ip_address = "192.168.1.100";
    int image_width = 8192;
    int image_height = 1024;
    int pixel_depth = 8;
    double exposure_us = 500.0;
    double line_rate_hz = 20000.0;
    int trigger_mode = 1;
    int buffer_count = 4;
};

struct TritConfig {
    std::string onnx_path = "models/yolov8n_defect.onnx";
    std::string engine_path = "models/yolov8n_defect.engine";
    int gpu_id = 0;
    int input_width = 640;
    int input_height = 640;
    int batch_size = 1;
    bool fp16 = true;
    bool int8 = false;
    std::string calib_cache = "";
    int max_workspace_mb = 2048;
    float score_threshold = 0.25f;
    float nms_threshold = 0.45f;
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

struct PipelineConfig {
    CameraConfig camera;
    TritConfig trt;
    TrackerConfig tracker;
    int ring_buffer_size = 8;
    bool save_result_image = false;
    std::string output_dir = "output/";
    int warmup_iterations = 50;
    bool use_simulator = true;
    std::string simulator_image_path = "";
};

PipelineConfig LoadConfig(const std::string& json_path);

}
