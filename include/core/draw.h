#pragma once
#include <opencv2/core.hpp>
cv::Mat sideBySide(const cv::Mat& A, const cv::Mat& B);
void     drawLine(cv::Mat& img, const cv::Vec3f& line, const cv::Scalar& color);

