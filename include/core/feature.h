#pragma once
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <vector>

void detectDescribeORB(const cv::Mat& gray, int num_features,
                       std::vector<cv::KeyPoint>& kps, cv::Mat& desc);

