#pragma once
#include <opencv2/core.hpp>
#include <vector>

struct RansacResult {
    cv::Mat F, E, inlierMask;
    std::vector<cv::Point2f> inL, inR;
    int inlierCount = 0;
};

RansacResult estimateGeometry(const std::vector<cv::Point2f>& ptsL,
                              const std::vector<cv::Point2f>& ptsR,
                              bool haveK, const cv::Mat& K,
                              double F_px_thresh, double F_conf,
                              double E_norm_thresh, double E_conf);

void drawEpipolarOverlays(const cv::Mat& img1, const cv::Mat& img2,
                          const std::vector<cv::Point2f>& inL,
                          const std::vector<cv::Point2f>& inR,
                          const cv::Mat& F, cv::Mat& out1, cv::Mat& out2);

