#pragma once
#include <opencv2/core.hpp>
#include "app/intrinsics.h"

// Step-by-step tutorial UI
void runTeach(const cv::Mat& leftColor, const cv::Mat& rightColor, const Intrinsics& Kinfo);

// Minimal GUI-only playground (same pipeline, no Learn text)
void runSandbox(const cv::Mat& leftColor, const cv::Mat& rightColor, const Intrinsics& Kinfo);
