// src/teach/teach.cpp
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <algorithm>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include "teach/teach.h"
#include "app/params.h"
#include "app/intrinsics.h"
#include "core/feature.h"
#include "core/matching.h"
#include "core/geometry.h"
#include "core/draw.h"
#include "ui/panel.h"

// ----------------------- Teaching state -----------------------
enum class TeachStep {
    Intro, ORBTheory, ORBVisual, ORBParams,
    MatchTheory, MatchViz, RansacTheory, Ransac,
    InliersViz, EpiTheory, EpilinesViz, WrapUp
};

static TeachStep nextStep(TeachStep s){
    return (s==TeachStep::WrapUp) ? s : static_cast<TeachStep>((int)s+1);
}
static TeachStep prevStep(TeachStep s){
    return (s==TeachStep::Intro) ? s : static_cast<TeachStep>((int)s-1);
}

struct TeachCtx {
    // images
    cv::Mat left, right, Lg, Rg;

    // parameters
    MatchParams  MP;
    RansacParams RP;
    Intrinsics   Kinfo;

    // features & matches
    std::vector<cv::KeyPoint> kpL, kpR;
    cv::Mat descL, descR;
    std::vector<std::vector<cv::DMatch>> knnLR, knnRL;
    std::vector<cv::DMatch> matches;

    // robust geometry
    RansacResult R; // F,E,inlierMask,inL,inR,inlierCount

    // UI
    UIState ui;
    TeachStep step = TeachStep::Intro;

    // recompute flags
    bool needFeatures=true, needMatches=true, needRansac=true;
    bool doSymmetric=false, autoRansac=true;
};

// ----------------------- Teaching text -----------------------
static std::vector<std::string> linesIntro(bool haveK){
    return {
        "Welcome! In this tutorial you will:",
        " 1) Detect ORB features on two views of the same scene.",
        " 2) Match features robustly (BF + KNN + Lowe ratio; optional symmetric check).",
        haveK ? " 3) Estimate Essential matrix E with RANSAC (normalized coords)." :
                " 3) Estimate Fundamental matrix F with RANSAC (pixel coords).",
        " 4) Visualize inliers and epipolar lines.",
        " 5) (If K is known) recover relative pose (R, t^).",
        "Use the sidebar: sliders, checkboxes, and buttons. Scroll this panel via wheel,",
        "by dragging the scrollbar thumb, or with j/k keys."
    };
}
static std::vector<std::string> linesORB(){
    return {
        "ORB = Oriented FAST + Rotated BRIEF:",
        " • FAST detects corners using a circle test around each pixel.",
        " • BRIEF builds a binary descriptor from many intensity comparisons.",
        " • ORB adds orientation so descriptors are rotation-invariant.",
        "Tips: more features → more candidates (and time). 2000–3000 works well near 1MP.",
    };
}
static std::vector<std::string> linesMatch(){
    return {
        "Matching pipeline:",
        " • KNN (k=2) using Hamming distance (binary descriptors).",
        " • Lowe ratio keeps a match if d1 < ratio * d2 (0.65–0.80 typical).",
        " • Symmetric (mutual) check: match must agree both directions.",
        "This removes many ambiguous repeats (e.g., windows)."
    };
}
static std::vector<std::string> linesRansac(bool haveK){
    return {
        "RANSAC: sample minimal set → estimate model → count inliers vs threshold.",
        haveK ? "For E (normalized coords), threshold is small (≈0.001–0.01)." :
                 "For F (pixels), threshold is a few pixels (≈0.5–3).",
        "Confidence controls number of trials; higher = more iterations.",
        "Workflow: clean matches (lower ratio, enable symmetric) then tighten threshold."
    };
}
static std::vector<std::string> linesEpi(bool haveK){
    return {
        "Epipolar geometry:",
        " • Pixel: x2^T F x1 = 0.  Normalized: x2'^T E x1' = 0.",
        "   x' = K^{-1} x and F = K^{-T} E K^{-1}.",
        " • For point x1 in image 1, corresponding x2 must lie on line l2 = F x1.",
        "Point-to-line distance (ax+by+c=0) is |ax+by+c|/sqrt(a^2+b^2).",
        haveK ? "We estimate E in normalized coords and visualize lines via F." :
                 "We estimate F directly and draw its epipolar lines."
    };
}
static std::vector<std::string> linesWrap(){
    return {
        "You built: ORB → matching → RANSAC F/E → inliers → epipolar lines.",
        "Polish for README: a short GIF, a figure with keypoints, matches, and lines,",
        "and a short table with inliers/ratio for different pairs.",
        "Next steps: try different image pairs; with K, show recovered (R,t^) as a sketch."
    };
}

// ----------------------- Pipeline helpers -----------------------
static void ensureFeatures(TeachCtx& C){
    if (!C.needFeatures) return;
    C.MP.num_features = std::clamp(C.MP.num_features, 200, 8000);

    detectDescribeORB(C.Lg, C.MP.num_features, C.kpL, C.descL);
    detectDescribeORB(C.Rg, C.MP.num_features, C.kpR, C.descR);

    C.needFeatures = false;
    C.needMatches  = true;
    C.needRansac   = true;

    // precompute KNN in both directions for optional symmetry
    C.knnLR.clear(); C.knnRL.clear();
    if (C.descL.empty() || C.descR.empty()) return;
    cv::BFMatcher m(cv::NORM_HAMMING,false);
    m.knnMatch(C.descL, C.descR, C.knnLR, 2);
    m.knnMatch(C.descR, C.descL, C.knnRL, 2);
}

static void ensureMatches(TeachCtx& C){
    if (!C.needMatches) return;
    auto one = ratioTestAndSort(C.knnLR, C.MP.ratio);
    if (C.doSymmetric) {
        auto back = ratioTestAndSort(C.knnRL, C.MP.ratio);
        C.matches = symmetricCheck(one, back);
    } else {
        C.matches = std::move(one);
    }
    C.needMatches = false;
    C.needRansac  = true;
}

static void ensureRansac(TeachCtx& C){
    if (!C.needRansac) return;

    std::vector<cv::Point2f> ptsL, ptsR;
    collectPoints(C.kpL, C.kpR, C.matches, ptsL, ptsR);

    C.R = estimateGeometry(ptsL, ptsR, C.Kinfo.haveK, C.Kinfo.K,
                           C.RP.F_px_thresh, C.RP.F_conf,
                           C.RP.E_norm_thresh, C.RP.E_conf);
    C.needRansac = false;
}

static void runRansacNow(TeachCtx& C){
    ensureFeatures(C);
    ensureMatches(C);
    C.needRansac = true;
    ensureRansac(C);
}

// ----------------------- UI population -----------------------
static void buildControlsForStep(TeachCtx& C, std::string& header, std::vector<std::string>& info){
    C.ui.sliders.clear(); C.ui.checks.clear(); C.ui.buttons.clear();

    auto addSlider = [&](std::string n, float mn,float mx,float v,bool asInt,std::string tip){
        Slider s; s.name=n; s.minv=mn; s.maxv=mx; s.val=v; s.isInt=asInt; s.tip=tip;
        C.ui.sliders.push_back(s);
    };
    auto addCheck = [&](std::string n, bool checked, std::string tip){
        Checkbox c; c.name=n; c.checked=checked; c.tip=tip; C.ui.checks.push_back(c);
    };
    auto addBtn = [&](std::string l, bool p=false){ Button b; b.label=l; b.primary=p; C.ui.buttons.push_back(b); };

    // Common buttons
    addBtn("Back"); addBtn("Next", true); addBtn("Exit");

    // Step-specific widgets & info
    switch (C.step) {
        case TeachStep::Intro:
            header = "Step: Intro";
            info   = linesIntro(C.Kinfo.haveK);
            break;

        case TeachStep::ORBTheory:
            header = "Step: ORB (theory)";
            info   = linesORB();
            break;

        case TeachStep::ORBVisual:
        case TeachStep::ORBParams: {
            header = "Step: ORB features";
            info   = {
                "Adjust the number of ORB features; keypoints update live in both images.",
                "Too few → unstable; too many → slower and potentially more ambiguous."
            };
            addSlider("features",  200.f, 8000.f, (float)C.MP.num_features, true,  "Max ORB keypoints per image.");
            addSlider("ratio",     0.50f, 0.95f, C.MP.ratio,               false, "Lowe ratio test. Lower = stricter.");
            addCheck ("symmetric", C.doSymmetric, "Keep only mutual best matches.");
        } break;

        case TeachStep::MatchTheory:
            header = "Step: Matching (theory)";
            info   = linesMatch();
            addSlider("ratio", 0.50f, 0.95f, C.MP.ratio, false, "Lowe ratio test.");
            addCheck ("symmetric", C.doSymmetric, "Mutual check.");
            break;

        case TeachStep::MatchViz: {
            header = "Step: Tentative matches";
            info   = { "These are tentative matches after ratio / symmetric filtering.",
                       "Limit how many to draw for clarity." };
            addSlider("ratio",     0.50f, 0.95f, C.MP.ratio, false, "Lowe ratio test.");
            addCheck ("symmetric", C.doSymmetric, "Mutual check.");
            addSlider("max draw",  10.f,  800.f,  (float)C.MP.max_draw, true, "Limit drawn matches.");
        } break;

        case TeachStep::RansacTheory:
            header = "Step: RANSAC (theory)";
            info   = linesRansac(C.Kinfo.haveK);
            break;

        case TeachStep::Ransac: {
            header = C.Kinfo.haveK ? "Step: Estimate E (RANSAC)" : "Step: Estimate F (RANSAC)";
            info   = { "Press 'Run RANSAC' for an immediate fit. Auto runs on param changes too.",
                       "Clean matches first (ratio, symmetric) then tighten threshold." };
            if (C.Kinfo.haveK) addSlider("E thr",     5e-4f, 2e-2f, (float)C.RP.E_norm_thresh, false, "Inlier threshold (normalized).");
            else               addSlider("F thr(px)", 0.5f,  5.0f,  (float)C.RP.F_px_thresh,   false, "Inlier threshold (pixels).");
            addSlider("conf", 0.90f, 0.999f, (float)(C.Kinfo.haveK ? C.RP.E_conf : C.RP.F_conf), false, "RANSAC confidence.");
            addBtn("Run RANSAC", true);
        } break;

        case TeachStep::InliersViz: {
            header = "Step: Inlier matches";
            info   = { "These matches agree with the estimated model (within threshold).",
                       "Green numbers show inlier count and percentage." };
            addSlider("max draw", 10.f, 800.f, (float)C.MP.max_draw, true, "Limit drawn matches.");
            addBtn("Run RANSAC"); // allow refresh
        } break;

        case TeachStep::EpiTheory:
            header = "Step: Epipolar lines (theory)";
            info   = linesEpi(C.Kinfo.haveK);
            break;

        case TeachStep::EpilinesViz: {
            header = "Step: Epipolar lines";
            info   = { "Each inlier point in one image maps to a line in the other.",
                       "The corresponding point should lie close to that line." };
            addBtn("Run RANSAC"); // quick refresh if params changed
        } break;

        case TeachStep::WrapUp:
            header = "Step: Wrap up";
            info   = linesWrap();
            break;
    }
}

// ----------------------- Docked frame composer -----------------------
// Reserve a left dock, clamp height & total width, and return the composed frame.
// Also outputs dockW_now so uiDockLeft can size the panels correctly.
static cv::Mat composeDockedFrame(UIState& UI, cv::Mat pair,
                                  int maxDisplayH, int maxWindowW,
                                  int& dockW_now) {
    // 1) Height cap
    double s = 1.0;
    if (pair.rows > maxDisplayH) {
        s = std::min(s, (double)maxDisplayH / (double)pair.rows);
    }

    // 2) Estimate dock width from font scale, clamp so images have room
    int dockW_est = std::max(320, (int)std::round(420 * uiGetGlobalFontScale()));
    const int marginW   = 12;    // small right margin
    const int minPairW  = 480;   // don't let the image pair get too tiny
    int maxDockW        = std::max(280, maxWindowW - minPairW - marginW);
    dockW_now           = std::clamp(dockW_est, 280, maxDockW);

    // 3) Width cap for the image pair so (dock + pair + margin) <= maxWindowW
    int maxPairW = std::max(200, maxWindowW - dockW_now - marginW);
    if (pair.cols > maxPairW) {
        s = std::min(s, (double)maxPairW / (double)pair.cols);
    }

    // 4) Apply scaling if needed
    if (s < 1.0) cv::resize(pair, pair, cv::Size(), s, s, cv::INTER_AREA);

    // 5) Build final frame (dock on left, images on right)
    int minDockHeight = std::max(700, (int)std::round(700 * uiGetGlobalFontScale()));
    int H_now         = std::max(pair.rows, minDockHeight);
    cv::Mat frame(H_now, dockW_now + pair.cols, CV_8UC3, cv::Scalar(25,25,25));

    // vertical separator
    cv::line(frame, cv::Point(dockW_now, 0), cv::Point(dockW_now, H_now-1), cv::Scalar(70,70,70), 2, cv::LINE_AA);

    // images on the right
    pair.copyTo(frame(cv::Rect(dockW_now, 0, pair.cols, std::min(pair.rows, H_now))));

    return frame;
}

// ----------------------- Core loop (teach or gui) -----------------------
static void runCoreUI(const cv::Mat& leftColor, const cv::Mat& rightColor, const Intrinsics& Kinfo, bool withTeachText){
    const std::string WIN = withTeachText ? "Teach" : "GUI";

    TeachCtx C;
    C.left = leftColor.clone(); C.right = rightColor.clone();
    cv::cvtColor(C.left,  C.Lg, cv::COLOR_BGR2GRAY);
    cv::cvtColor(C.right, C.Rg, cv::COLOR_BGR2GRAY);
    C.Kinfo = Kinfo;

    // defaults
    C.MP.num_features = 2500;
    C.MP.ratio        = 0.75f;
    C.MP.max_draw     = 150;
    C.doSymmetric     = false;
    if (C.Kinfo.haveK) { C.RP.E_norm_thresh = 0.0015; C.RP.E_conf = 0.999; }
    else               { C.RP.F_px_thresh   = 1.0;    C.RP.F_conf = 0.999; }

    // initial compute
    ensureFeatures(C);
    ensureMatches(C);
    ensureRansac(C);

    // window + mouse
    cv::namedWindow(WIN, cv::WINDOW_NORMAL);
    cv::setMouseCallback(WIN, uiHandleMouse, &C.ui);
    uiInit(C.ui, {C.left.cols + C.right.cols + 520, std::max(C.left.rows, C.right.rows)});
    uiSetGlobalFontScale(1.15, &C.ui); // comfy default

    while (true) {
        // recompute as needed
        ensureFeatures(C);
        ensureMatches(C);
        if (C.autoRansac) ensureRansac(C);

        // build right-side imagery
        cv::Mat pair;
        switch (C.step) {
            case TeachStep::ORBVisual:
            case TeachStep::ORBParams: {
                cv::Mat Lk,Rk;
                cv::drawKeypoints(C.left,  C.kpL, Lk, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
                cv::drawKeypoints(C.right, C.kpR, Rk, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
                pair = sideBySide(Lk,Rk);
            } break;
            case TeachStep::MatchViz: {
                int drawN = std::min(C.MP.max_draw, (int)C.matches.size());
                std::vector<cv::DMatch> sub(C.matches.begin(), C.matches.begin()+drawN);
                cv::drawMatches(C.left, C.kpL, C.right, C.kpR, sub, pair);
            } break;
            case TeachStep::Ransac:
            case TeachStep::InliersViz: {
                // draw inlier matches (if available)
                std::vector<cv::DMatch> inlierMatches;
                if (!C.R.inlierMask.empty()) {
                    for (size_t i=0;i<C.matches.size();++i)
                        if (C.R.inlierMask.at<uchar>((int)i)) inlierMatches.push_back(C.matches[i]);
                }
                int drawN = std::min(C.MP.max_draw, (int)inlierMatches.size());
                std::vector<cv::DMatch> sub(inlierMatches.begin(), inlierMatches.begin()+drawN);
                cv::drawMatches(C.left, C.kpL, C.right, C.kpR, sub, pair);
            } break;
            case TeachStep::EpilinesViz: {
                cv::Mat e1,e2; drawEpipolarOverlays(C.left, C.right, C.R.inL, C.R.inR, C.R.F, e1, e2);
                pair = sideBySide(e1,e2);
            } break;
            default:
                pair = sideBySide(C.left, C.right);
        }

        // compose docked frame (panels left, images right)
        int dockW_now = 0;
        const int kMaxDisplayH = 900;    // adjust to your screen height
        const int kMaxWindowW  = 1360;   // adjust to your screen width
        cv::Mat frame = composeDockedFrame(C.ui, pair, kMaxDisplayH, kMaxWindowW, dockW_now);

        // lay out the panels into the dock area
        uiUpdateDockBounds(C.ui, /*x=*/0, /*width=*/dockW_now, /*height=*/frame.rows);


        // build controls and lesson text for this step
        std::string header; std::vector<std::string> info;
        buildControlsForStep(C, header, info);

        // metrics strip content
        C.ui.metrics.clear();
        {
            using M = Metric;
            int kpL = (int)C.kpL.size(), kpR = (int)C.kpR.size();
            int tent = (int)C.matches.size();
            int inl  = (int)C.R.inlierCount;
            double ratio = (tent>0) ? (100.0 * inl / tent) : 0.0;

            auto fmtInt = [](int v){ return std::to_string(v); };
            auto fmtPct = [](double p){ char b[32]; std::snprintf(b,sizeof(b),"%.1f%%",p); return std::string(b); };

            if (C.step==TeachStep::ORBVisual || C.step==TeachStep::ORBParams) {
                C.ui.metrics.push_back(M{"Keypoints (L/R):", fmtInt(kpL)+" / "+fmtInt(kpR), {80,180,255}, 0.75});
            }
            if (C.step==TeachStep::MatchViz || C.step==TeachStep::Ransac || C.step==TeachStep::InliersViz || C.step==TeachStep::EpilinesViz) {
                C.ui.metrics.push_back(M{"Keypoints (L/R):", fmtInt(kpL)+" / "+fmtInt(kpR), {200,200,200}, 0.60});
                C.ui.metrics.push_back(M{"Tentative matches:", fmtInt(tent),               {255,210,100}, 0.75});
            }
            if (C.step==TeachStep::Ransac || C.step==TeachStep::InliersViz || C.step==TeachStep::EpilinesViz) {
                C.ui.metrics.push_back(M{"RANSAC inliers:", fmtInt(inl)+"  ("+fmtPct(ratio)+")", {120,255,120}, 0.85});
                if (C.Kinfo.haveK)
                    C.ui.metrics.push_back(M{"E thr / conf:",  std::to_string(C.RP.E_norm_thresh)+" / "+std::to_string(C.RP.E_conf), {180,220,255}, 0.60});
                else
                    C.ui.metrics.push_back(M{"F thr(px) / conf:",  std::to_string(C.RP.F_px_thresh)+" / "+std::to_string(C.RP.F_conf), {180,220,255}, 0.60});
            }
        }

        // draw panels and (optionally) the Learn text
        uiDrawControls(frame, C.ui, header);
        if (withTeachText) uiDrawInfo(frame, C.ui, "Learn", info);

        // show composed frame only
        cv::imshow(WIN, frame);
        int key = cv::waitKey(20);

        // zoom + keyboard scroll
        if (key=='+' || key=='=') uiSetGlobalFontScale(std::min(2.0, uiGetGlobalFontScale()*1.10), &C.ui);
        if (key=='-' || key=='_') uiSetGlobalFontScale(std::max(0.6, uiGetGlobalFontScale()/1.10), &C.ui);
        if (key=='j' || key=='J') C.ui.infoPanel.scrollOffset += 3;
        if (key=='k' || key=='K') C.ui.infoPanel.scrollOffset -= 3;

        // apply slider/checkbox changes (clamped)
        for (auto& s : C.ui.sliders) {
            if (s.name=="features") {
                int nf = (int)std::lround(std::clamp(s.val, 200.f, 8000.f));
                if (nf!=C.MP.num_features){ C.MP.num_features=nf; C.needFeatures=true; }
            } else if (s.name=="ratio") {
                float r = std::clamp(s.val, 0.50f, 0.95f);
                if (r!=C.MP.ratio){ C.MP.ratio=r; C.needMatches=true; }
            } else if (s.name=="max draw") {
                int md = (int)std::lround(std::clamp(s.val, 10.f, 800.f));
                C.MP.max_draw = md;
            } else if (s.name=="F thr(px)") {
                C.RP.F_px_thresh = std::clamp((double)s.val, 0.5, 5.0); C.needRansac=true;
            } else if (s.name=="E thr") {
                C.RP.E_norm_thresh = std::clamp((double)s.val, 5e-4, 2e-2); C.needRansac=true;
            } else if (s.name=="conf") {
                double c = std::clamp((double)s.val, 0.90, 0.999);
                if (C.Kinfo.haveK) C.RP.E_conf=c; else C.RP.F_conf=c; C.needRansac=true;
            }
        }
        for (auto& c : C.ui.checks) {
            if (c.name=="symmetric") {
                if (c.checked!=C.doSymmetric){ C.doSymmetric=c.checked; C.needMatches=true; }
            }
        }

        // buttons
        if (C.ui.clickedButton != -1 && C.ui.clickedButton < (int)C.ui.buttons.size()) {
            std::string lab = C.ui.buttons[C.ui.clickedButton].label;
            if (lab=="Exit") break;
            else if (lab=="Back") C.step = prevStep(C.step);
            else if (lab=="Next") C.step = nextStep(C.step);
            else if (lab=="Run RANSAC") { C.autoRansac=false; runRansacNow(C); C.step = TeachStep::InliersViz; }
            C.ui.clickedButton = -1;
        }

        // quit keys
        if (key==27 || key=='q' || key=='Q') break;
    }
}

// ----------------------- Public entry points -----------------------
void runTeach(const cv::Mat& leftColor, const cv::Mat& rightColor, const Intrinsics& Kinfo){
    runCoreUI(leftColor, rightColor, Kinfo, /*withTeachText=*/true);
}
void runSandbox(const cv::Mat& leftColor, const cv::Mat& rightColor, const Intrinsics& Kinfo){
    runCoreUI(leftColor, rightColor, Kinfo, /*withTeachText=*/false);
}
