#include "core/geometry.h"
#include "core/draw.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

RansacResult estimateGeometry(const std::vector<cv::Point2f>& ptsL,
                              const std::vector<cv::Point2f>& ptsR,
                              bool haveK, const cv::Mat& K,
                              double F_px_thresh, double F_conf,
                              double E_norm_thresh, double E_conf) {
    RansacResult R;
    if (ptsL.size() < 8 || ptsR.size() < 8) return R;

    if (haveK) {
        R.E = cv::findEssentialMat(ptsL, ptsR, K, cv::RANSAC, E_conf, E_norm_thresh, R.inlierMask);
        if (!R.E.empty()) {
            cv::Mat Kinv = K.inv();
            R.F = Kinv.t() * R.E * Kinv;
        }
    } else {
        R.F = cv::findFundamentalMat(ptsL, ptsR, cv::FM_RANSAC, F_px_thresh, F_conf, R.inlierMask);
    }
    if (R.inlierMask.empty()) return R;

    R.inlierCount = 0;
    for (int i = 0; i < R.inlierMask.rows; ++i) {
        if (R.inlierMask.at<uchar>(i)) { R.inlierCount++; R.inL.push_back(ptsL[i]); R.inR.push_back(ptsR[i]); }
    }
    return R;
}

void drawEpipolarOverlays(const cv::Mat& img1, const cv::Mat& img2,
                          const std::vector<cv::Point2f>& inL,
                          const std::vector<cv::Point2f>& inR,
                          const cv::Mat& F, cv::Mat& out1, cv::Mat& out2) {
    out1 = img1.clone(); out2 = img2.clone();
    if (inL.empty() || inR.empty() || F.empty()) return;

    std::vector<cv::Vec3f> lines1, lines2;
    cv::computeCorrespondEpilines(inR, 2, F, lines1);
    cv::computeCorrespondEpilines(inL, 1, F, lines2);

    for (size_t i=0;i<inL.size();++i) {
        drawLine(out2, lines2[i], {60,220,60});
        cv::circle(out2, inR[i], 3, {30,30,230}, -1, cv::LINE_AA);
        drawLine(out1, lines1[i], {60,220,60});
        cv::circle(out1, inL[i], 3, {30,30,230}, -1, cv::LINE_AA);
    }
}

