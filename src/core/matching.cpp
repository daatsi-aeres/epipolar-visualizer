#include "core/matching.h"
#include <algorithm>

std::vector<cv::DMatch> ratioTestAndSort(
    const std::vector<std::vector<cv::DMatch>>& knn, float ratio) {
    std::vector<cv::DMatch> good; good.reserve(knn.size());
    for (const auto& v: knn) {
        if (v.size() < 2) continue;
        const auto& m = v[0]; const auto& n = v[1];
        if (m.distance < ratio * n.distance) good.push_back(m);
    }
    std::sort(good.begin(), good.end(),
              [](const cv::DMatch& a, const cv::DMatch& b){ return a.distance < b.distance; });
    return good;
}

std::vector<cv::DMatch> symmetricCheck(
    const std::vector<cv::DMatch>& LtoR,
    const std::vector<cv::DMatch>& RtoL) {
    std::vector<int> r2l(100000, -1); int mx=0;
    for (const auto& m: RtoL) {
        if (m.queryIdx >= (int)r2l.size()) continue;
        r2l[m.queryIdx] = m.trainIdx;
        mx = std::max({mx, m.queryIdx, m.trainIdx});
    }
    if ((int)r2l.size() > mx+5) r2l.resize(mx+5, -1);

    std::vector<cv::DMatch> out; out.reserve(std::min(LtoR.size(), RtoL.size()));
    for (const auto& m: LtoR)
        if (m.trainIdx < (int)r2l.size() && r2l[m.trainIdx] == m.queryIdx) out.push_back(m);

    std::sort(out.begin(), out.end(),
              [](const cv::DMatch& a, const cv::DMatch& b){ return a.distance < b.distance; });
    return out;
}

void collectPoints(const std::vector<cv::KeyPoint>& kpL,
                   const std::vector<cv::KeyPoint>& kpR,
                   const std::vector<cv::DMatch>& matches,
                   std::vector<cv::Point2f>& ptsL,
                   std::vector<cv::Point2f>& ptsR) {
    ptsL.clear(); ptsR.clear();
    ptsL.reserve(matches.size()); ptsR.reserve(matches.size());
    for (const auto& m : matches) {
        ptsL.push_back(kpL[m.queryIdx].pt);
        ptsR.push_back(kpR[m.trainIdx].pt);
    }
}

