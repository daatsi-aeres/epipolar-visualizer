#include "core/draw.h"
#include <opencv2/imgproc.hpp>

cv::Mat sideBySide(const cv::Mat& A, const cv::Mat& B) {
    int h = std::max(A.rows, B.rows);
    cv::Mat side(h, A.cols + B.cols, A.type(), cv::Scalar::all(0));
    A.copyTo(side(cv::Rect(0,0,A.cols,A.rows)));
    B.copyTo(side(cv::Rect(A.cols,0,B.cols,B.rows)));
    cv::line(side, {A.cols,0}, {A.cols,h-1}, {200,200,200}, 2);
    return side;
}

void drawLine(cv::Mat& img, const cv::Vec3f& line, const cv::Scalar& color) {
    const float a=line[0], b=line[1], c=line[2];
    int w=img.cols, h=img.rows;
    cv::Point p1,p2;
    if (std::fabs(b) > 1e-6f) {
        p1 = {0, (int)std::lround(-c/b)};
        p2 = {w-1, (int)std::lround(-(c + a*(w-1))/b)};
    } else {
        int x = (int)std::lround(-c/a);
        p1 = {x,0}; p2 = {x,h-1};
    }
    cv::line(img, p1, p2, color, 1, cv::LINE_AA);
}

