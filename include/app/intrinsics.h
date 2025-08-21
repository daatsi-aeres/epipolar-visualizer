#pragma once
#include <opencv2/core.hpp>

struct Intrinsics {
    bool   haveK = false;
    cv::Mat K;
    double fx=0, fy=0, cx=0, cy=0;
};

