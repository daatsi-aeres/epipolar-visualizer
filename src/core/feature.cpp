#include "core/feature.h"
void detectDescribeORB(const cv::Mat& gray, int num_features,
                       std::vector<cv::KeyPoint>& kps, cv::Mat& desc) {
    auto orb = cv::ORB::create(num_features);
    orb->detectAndCompute(gray, cv::noArray(), kps, desc);
}

