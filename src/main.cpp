#include <chrono>
#include <csignal>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>

#include "config.h"
#include "pipeline.h"

static volatile std::sig_atomic_t g_signal = 0;

void SignalHandler(int signum) {
    g_signal = 1;
    std::cout << "\n[Main] Received signal " << signum << ", shutting down...\n";
}

int main(int argc, char* argv[]) {
    std::signal(SIGINT, SignalHandler);
    std::signal(SIGTERM, SignalHandler);

    std::string config_path = "configs/default.json";
    if (argc > 1) {
        config_path = argv[1];
    }

    std::cout << "======================================================\n";
    std::cout << "  Steel Surface Defect Vision Pipeline\n";
    std::cout << "  Cold-rolled steel line-scan inspection system\n";
    std::cout << "  Target: <15ms end-to-end latency @ 20m/s\n";
    std::cout << "======================================================\n\n";

    auto cfg = steel::LoadConfig(config_path);

    std::cout << "[Main] Config loaded from: " << config_path << "\n";
    std::cout << "[Main] Camera: " << cfg.camera.camera_type
              << " " << cfg.camera.image_width << "x" << cfg.camera.image_height
              << " @ " << cfg.camera.line_rate_hz << " Hz\n";
    std::cout << "[Main] TRT Engine: " << cfg.trt.onnx_path
              << " -> " << cfg.trt.engine_path
              << " (FP16=" << cfg.trt.fp16 << " INT8=" << cfg.trt.int8 << ")\n";
    std::cout << "[Main] Score thresh: " << cfg.trt.score_threshold
              << " NMS thresh: " << cfg.trt.nms_threshold << "\n\n";

    steel::Pipeline pipeline(cfg);

    if (!pipeline.Init()) {
        std::cerr << "[Main] Pipeline init failed\n";
        return 1;
    }

    if (!pipeline.Start()) {
        std::cerr << "[Main] Pipeline start failed\n";
        return 1;
    }

    std::cout << "[Main] Pipeline running. Press Ctrl+C to stop.\n\n";

    while (g_signal == 0 && pipeline.IsRunning()) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    pipeline.Stop();

    auto stats = pipeline.GetStats();
    std::cout << "\n[Main] ============ Final Statistics ============\n";
    std::cout << "[Main] Total frames:  " << stats.total_frames << "\n";
    std::cout << "[Main] Defect frames: " << stats.defect_frames << "\n";
    if (stats.total_frames > 0) {
        std::cout << "[Main] Defect rate:   "
                  << std::fixed << std::setprecision(2)
                  << (100.0 * stats.defect_frames / stats.total_frames) << "%\n";
    }
    std::cout << "[Main] Avg E2E:       " << std::fixed << std::setprecision(3)
              << stats.avg_e2e_ms << " ms\n";
    std::cout << "[Main]   Capture:     " << stats.avg_capture_ms << " ms\n";
    std::cout << "[Main]   Preprocess:  " << stats.avg_preprocess_ms << " ms\n";
    std::cout << "[Main]   Inference:   " << stats.avg_infer_ms << " ms\n";
    std::cout << "[Main]   Postprocess: " << stats.avg_postprocess_ms << " ms\n";

    if (stats.avg_e2e_ms <= 15.0 && stats.total_frames > 0) {
        std::cout << "[Main] *** LATENCY TARGET <15ms ACHIEVED ***\n";
    } else if (stats.total_frames > 0) {
        std::cout << "[Main] *** WARNING: Latency target NOT met ***\n";
    }
    std::cout << "[Main] ==========================================\n";

    return 0;
}
