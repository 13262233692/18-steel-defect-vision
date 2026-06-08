#include "yolo_postprocess.h"

#include <opencv2/opencv.hpp>

namespace steel {

void DrawDetections(
    cv::Mat& image,
    const std::vector<Detection>& dets,
    const std::vector<std::string>& class_names)
{
    static const cv::Scalar colors[] = {
        {0, 0, 255}, {0, 255, 0}, {255, 0, 0},
        {0, 255, 255}, {255, 0, 255}, {255, 255, 0},
        {128, 0, 255}, {0, 128, 255}, {255, 128, 0}
    };

    for (size_t i = 0; i < dets.size(); ++i) {
        const auto& d = dets[i];
        cv::Point pt1(static_cast<int>(d.x1), static_cast<int>(d.y1));
        cv::Point pt2(static_cast<int>(d.x2), static_cast<int>(d.y2));
        cv::Scalar color = colors[d.class_id % 9];

        cv::rectangle(image, pt1, pt2, color, 2);

        std::string label;
        if (d.class_id < static_cast<int>(class_names.size())) {
            label = class_names[d.class_id];
        } else {
            label = "cls" + std::to_string(d.class_id);
        }

        if (d.track_id >= 0) {
            label = "#" + std::to_string(d.track_id) + " " + label;
        }

        char conf_buf[16];
        snprintf(conf_buf, sizeof(conf_buf), " %.1f%%", d.confidence * 100.f);
        label += conf_buf;

        int baseline = 0;
        double font_scale = 0.5;
        int thickness = 1;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
            font_scale, thickness, &baseline);

        int text_y = static_cast<int>(d.y1) - 4;
        if (text_y < text_size.height + 4) {
            text_y = static_cast<int>(d.y2) + text_size.height + 4;
        }

        cv::Point text_org(static_cast<int>(d.x1), text_y);
        cv::rectangle(image,
            cv::Point(text_org.x, text_org.y - text_size.height - 2),
            cv::Point(text_org.x + text_size.width, text_org.y + 2),
            color, cv::FILLED);
        cv::putText(image, label, text_org, cv::FONT_HERSHEY_SIMPLEX,
            font_scale, cv::Scalar(255, 255, 255), thickness);
    }
}

std::vector<std::string> DefaultDefectClassNames() {
    return {
        "scratch",
        "oxide_scale",
        "sand_hole",
        "crack",
        "pit",
        "roll_mark",
        "bubble",
        "inclusion",
        "stain"
    };
}

}
