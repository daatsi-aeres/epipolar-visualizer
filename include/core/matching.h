#pragma once
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <vector>

std::vector<cv::DMatch> ratioTestAndSort(
    const std::vector<std::vector<cv::DMatch>>& knn, float ratio);

std::vector<cv::DMatch> symmetricCheck(
    const std::vector<cv::DMatch>& LtoR,
    const std::vector<cv::DMatch>& RtoL);

void collectPoints(const std::vector<cv::KeyPoint>& kpL,
                   const std::vector<cv::KeyPoint>& kpR,
                   const std::vector<cv::DMatch>& matches,
                   std::vector<cv::Point2f>& ptsL,
                   std::vector<cv::Point2f>& ptsR);

