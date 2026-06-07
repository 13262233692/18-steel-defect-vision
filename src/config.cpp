#include "config.h"

#include <fstream>
#include <iostream>

#include <nlohmann/json.hpp>

namespace steel {

PipelineConfig LoadConfig(const std::string& json_path) {
    PipelineConfig cfg;

    std::ifstream fin(json_path);
    if (!fin.is_open()) {
        std::cerr << "[Config] Cannot open " << json_path << ", using defaults\n";
        return cfg;
    }

    try {
        nlohmann::json j;
        fin >> j;

        if (j.contains("camera")) {
            auto& c = j["camera"];
            if (c.contains("camera_type")) cfg.camera.camera_type = c["camera_type"].get<std::string>();
            if (c.contains("ip_address")) cfg.camera.ip_address = c["ip_address"].get<std::string>();
            if (c.contains("image_width")) cfg.camera.image_width = c["image_width"].get<int>();
            if (c.contains("image_height")) cfg.camera.image_height = c["image_height"].get<int>();
            if (c.contains("pixel_depth")) cfg.camera.pixel_depth = c["pixel_depth"].get<int>();
            if (c.contains("exposure_us")) cfg.camera.exposure_us = c["exposure_us"].get<double>();
            if (c.contains("line_rate_hz")) cfg.camera.line_rate_hz = c["line_rate_hz"].get<double>();
            if (c.contains("trigger_mode")) cfg.camera.trigger_mode = c["trigger_mode"].get<int>();
            if (c.contains("buffer_count")) cfg.camera.buffer_count = c["buffer_count"].get<int>();
        }

        if (j.contains("trt")) {
            auto& t = j["trt"];
            if (t.contains("onnx_path")) cfg.trt.onnx_path = t["onnx_path"].get<std::string>();
            if (t.contains("engine_path")) cfg.trt.engine_path = t["engine_path"].get<std::string>();
            if (t.contains("gpu_id")) cfg.trt.gpu_id = t["gpu_id"].get<int>();
            if (t.contains("input_width")) cfg.trt.input_width = t["input_width"].get<int>();
            if (t.contains("input_height")) cfg.trt.input_height = t["input_height"].get<int>();
            if (t.contains("batch_size")) cfg.trt.batch_size = t["batch_size"].get<int>();
            if (t.contains("fp16")) cfg.trt.fp16 = t["fp16"].get<bool>();
            if (t.contains("int8")) cfg.trt.int8 = t["int8"].get<bool>();
            if (t.contains("calib_cache")) cfg.trt.calib_cache = t["calib_cache"].get<std::string>();
            if (t.contains("max_workspace_mb")) cfg.trt.max_workspace_mb = t["max_workspace_mb"].get<int>();
            if (t.contains("score_threshold")) cfg.trt.score_threshold = t["score_threshold"].get<float>();
            if (t.contains("nms_threshold")) cfg.trt.nms_threshold = t["nms_threshold"].get<float>();
        }

        if (j.contains("pipeline")) {
            auto& p = j["pipeline"];
            if (p.contains("ring_buffer_size")) cfg.ring_buffer_size = p["ring_buffer_size"].get<int>();
            if (p.contains("save_result_image")) cfg.save_result_image = p["save_result_image"].get<bool>();
            if (p.contains("output_dir")) cfg.output_dir = p["output_dir"].get<std::string>();
            if (p.contains("warmup_iterations")) cfg.warmup_iterations = p["warmup_iterations"].get<int>();
            if (p.contains("use_simulator")) cfg.use_simulator = p["use_simulator"].get<bool>();
            if (p.contains("simulator_image_path")) cfg.simulator_image_path = p["simulator_image_path"].get<std::string>();
        }

    } catch (const nlohmann::json::exception& e) {
        std::cerr << "[Config] JSON parse error: " << e.what() << "\n";
    }

    return cfg;
}

}
