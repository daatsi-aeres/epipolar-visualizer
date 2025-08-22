// Step 0: UI skeleton and layout. No algorithms yet.
// Next steps will add image loading, feature detection, matching, and epipolar geometry.

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <iostream>
#include <optional>
#include <chrono>
extern "C" {
#include "tinyfiledialogs.h"
}
#include <opencv2/flann/miniflann.hpp>


struct MatchResult {
    std::vector<cv::DMatch> matches;
    double match_ms = 0.0;
    cv::Mat vis; // visualization
};

// Simple container for one image pipeline state
struct ImageSlot {
    std::string path;
    cv::Mat original;   // BGR
    cv::Mat gray;       // 8-bit single channel
    std::vector<cv::KeyPoint> kpts;
    cv::Mat desc;       // ORB descriptors (CV_8U)
    cv::Mat vis;        // rendered preview for the slot size
    double detect_ms = 0.0;
    bool valid() const { return !original.empty(); }
};

// Fit src into dst rect keeping aspect, letterbox with dark gray
static cv::Mat letterbox(const cv::Mat& src, const cv::Size& dstSize) {
    cv::Mat out(dstSize, CV_8UC3, cv::Scalar(25,25,25));
    if (src.empty()) return out;
    double r = std::min(dstSize.width / (double)src.cols, dstSize.height / (double)src.rows);
    int w = std::max(1, (int)std::round(src.cols * r));
    int h = std::max(1, (int)std::round(src.rows * r));
    cv::Mat resized;
    cv::resize(src, resized, {w, h}, 0, 0, cv::INTER_AREA);
    int x = (dstSize.width - w) / 2;
    int y = (dstSize.height - h) / 2;
    resized.copyTo(out(cv::Rect(x, y, w, h)));
    return out;
}

// Draw a small overlay text block in a slot (top-left)
static void put_info(cv::Mat& slotBgr, int x, int y, const std::string& line) {
    cv::putText(slotBgr, line, {x, y}, cv::FONT_HERSHEY_SIMPLEX, 0.52, cv::Scalar(230,230,230), 1, cv::LINE_AA);
}
struct EpiResult {
    cv::Mat F, E, R, t;     // F always set; E/R/t only if intrinsics used
    int inliers = 0, total = 0;
    double ransac_ms = 0.0;
    bool usedEssential = false;
    cv::Mat visLeft, visRight; // letterboxed images with epilines
};

// Clip a line ax + by + c = 0 to the image rectangle [0..w-1]x[0..h-1].
static bool clip_line_to_image(const cv::Vec3f& L, int w, int h, cv::Point2f& p1, cv::Point2f& p2) {
    const double a = L[0], b = L[1], c = L[2];
    std::vector<cv::Point2f> pts;

    auto add_if_valid = [&](double x, double y){
        if (x >= 0 && x <= w-1 && y >= 0 && y <= h-1) pts.emplace_back((float)x,(float)y);
    };

    const double eps = 1e-9;

    // Intersections with x = 0 and x = w-1
    if (std::abs(b) > eps) {
        add_if_valid(0, -(c)/b);
        add_if_valid(w-1, -(a*(w-1)+c)/b);
    }
    // Intersections with y = 0 and y = h-1
    if (std::abs(a) > eps) {
        add_if_valid(-(c)/a, 0);
        add_if_valid(-(b*(h-1)+c)/a, h-1);
    }

    // Pick two distinct points
    if (pts.size() < 2) return false;
    // Remove duplicates if any
    if (cv::norm(pts[0] - pts[1]) < 1e-3) {
        if (pts.size() >= 3) p2 = pts[2];
        else return false;
    } else p2 = pts[1];
    p1 = pts[0];
    return true;
}

static cv::Mat draw_epilines_letterboxed(
    const cv::Mat& imgBgr,
    const std::vector<cv::Vec3f>& lines,  // ax+by+c=0 in this image's coords
    const cv::Size& slotSize,
    int max_lines = 80
) {
    cv::Mat out = letterbox(imgBgr, slotSize);
    if (imgBgr.empty() || lines.empty()) return out;

    // Map original->letterbox
    double r = std::min(slotSize.width / (double)imgBgr.cols, slotSize.height / (double)imgBgr.rows);
    int w = (int)std::round(imgBgr.cols * r);
    int h = (int)std::round(imgBgr.rows * r);
    int xoff = (slotSize.width - w) / 2;
    int yoff = (slotSize.height - h) / 2;

    const int W = imgBgr.cols, H = imgBgr.rows;
    int N = std::min<int>((int)lines.size(), max_lines);
    for (int i = 0; i < N; ++i) {
        cv::Point2f p1, p2;
        if (!clip_line_to_image(lines[i], W, H, p1, p2)) continue;
        // Map to letterbox coords
        cv::Point P1( xoff + (int)std::round(p1.x * r), yoff + (int)std::round(p1.y * r) );
        cv::Point P2( xoff + (int)std::round(p2.x * r), yoff + (int)std::round(p2.y * r) );
        cv::line(out, P1, P2, cv::Scalar(60,200,200), 1, cv::LINE_AA);
    }
    return out;
}

static EpiResult compute_epi_and_render(
    const ImageSlot& left, const ImageSlot& right,
    const std::vector<cv::DMatch>& matches,
    bool use_intrinsics, double fx, double fy, double cx, double cy,
    double ransac_thresh_px, double ransac_conf,
    const cv::Size& leftSlot, const cv::Size& rightSlot
) {
    EpiResult Rz;
    Rz.total = (int)matches.size();

    if (!left.valid() || !right.valid() || matches.size() < 8) {
        // Not enough matches for F (needs >=8). Just letterbox images.
        Rz.visLeft = letterbox(left.original, leftSlot);
        Rz.visRight = letterbox(right.original, rightSlot);
        return Rz;
    }

    // Build matched point sets (pixel coords)
    std::vector<cv::Point2f> pts1, pts2;
    pts1.reserve(matches.size());
    pts2.reserve(matches.size());
    for (const auto& m : matches) {
        pts1.push_back(left.kpts[m.queryIdx].pt);
        pts2.push_back(right.kpts[m.trainIdx].pt);
    }

    cv::Mat mask; // inliers from RANSAC
    auto t0 = std::chrono::high_resolution_clock::now();

    if (use_intrinsics && fx > 0 && fy > 0) {
        // Essential with intrinsics
        cv::Mat K = (cv::Mat_<double>(3,3) << fx,0,cx, 0,fy,cy, 0,0,1);
        Rz.E = cv::findEssentialMat(pts1, pts2, K, cv::RANSAC, ransac_conf, ransac_thresh_px, mask);
        Rz.usedEssential = !Rz.E.empty();
        if (Rz.usedEssential) {
            // Derive F = K^{-T} E K^{-1} (for epilines in pixel domain)
            cv::Mat Kinv = K.inv();
            Rz.F = Kinv.t() * Rz.E * Kinv;
            // Recover pose (optional nice-to-have)
            cv::recoverPose(Rz.E, pts1, pts2, K, Rz.R, Rz.t, mask);
        }
    } else {
        // Fundamental only
        Rz.F = cv::findFundamentalMat(pts1, pts2, cv::FM_RANSAC, ransac_thresh_px, ransac_conf, mask);
        Rz.usedEssential = false;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    Rz.ransac_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (Rz.F.empty() || mask.empty()) {
        // Could not estimate; just show images
        Rz.visLeft = letterbox(left.original, leftSlot);
        Rz.visRight = letterbox(right.original, rightSlot);
        return Rz;
    }

    // Inlier points only
    std::vector<cv::Point2f> in1, in2;
    in1.reserve(matches.size());
    in2.reserve(matches.size());
    for (int i = 0; i < mask.rows; ++i) {
        if (mask.at<uchar>(i)) {
            in1.push_back(pts1[i]);
            in2.push_back(pts2[i]);
        }
    }
    Rz.inliers = (int)in1.size();

    // Compute epilines: pts1 -> lines in right; pts2 -> lines in left
    cv::Mat lines2_, lines1_;
    if (!in1.empty()) cv::computeCorrespondEpilines(in1, 1, Rz.F, lines2_);
    if (!in2.empty()) cv::computeCorrespondEpilines(in2, 2, Rz.F, lines1_);

    std::vector<cv::Vec3f> lines2, lines1;
    lines2.assign((cv::Vec3f*)lines2_.datastart, (cv::Vec3f*)lines2_.dataend);
    lines1.assign((cv::Vec3f*)lines1_.datastart, (cv::Vec3f*)lines1_.dataend);

    // Render letterboxed views with epilines
    Rz.visLeft  = draw_epilines_letterboxed(left.original,  lines1, leftSlot);
    Rz.visRight = draw_epilines_letterboxed(right.original, lines2, rightSlot);

    return Rz;
}



static MatchResult match_features(const ImageSlot& left, const ImageSlot& right,
                                  bool use_flann, double ratio_test, bool cross_check,
                                  const cv::Size& slotSize)
{
    MatchResult res;
    if (!left.valid() || !right.valid() || left.desc.empty() || right.desc.empty()) {
        res.vis = letterbox(cv::Mat(), slotSize);
        return res;
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    if (use_flann) {
        // FLANN path: convert ORB (CV_8U) to CV_32F and do KNN+ratio.
        cv::Mat descL32f, descR32f;
        left.desc.convertTo(descL32f, CV_32F);
        right.desc.convertTo(descR32f, CV_32F);

        cv::Ptr<cv::DescriptorMatcher> matcher = cv::FlannBasedMatcher::create();
        std::vector<std::vector<cv::DMatch>> knn;
        matcher->knnMatch(descL32f, descR32f, knn, 2);

        for (auto& k : knn) {
            if (k.size() >= 2 && k[0].distance < ratio_test * k[1].distance) {
                res.matches.push_back(k[0]);
            }
        }
        // Note: cross_check is ignored for FLANN.
    } else {
        // BF + Hamming for ORB
        cv::BFMatcher bf(cv::NORM_HAMMING, cross_check);

        if (cross_check) {
            // crossCheck=true implies mutual best matches; KNN+ratio is NOT supported here.
            bf.match(left.desc, right.desc, res.matches);
        } else if (ratio_test < 0.99) {
            // Standard Lowe's ratio test
            std::vector<std::vector<cv::DMatch>> knn;
            bf.knnMatch(left.desc, right.desc, knn, 2);
            for (auto& k : knn) {
                if (k.size() >= 2 && k[0].distance < ratio_test * k[1].distance) {
                    res.matches.push_back(k[0]);
                }
            }
        } else {
            // No ratio test; just take best single matches
            bf.match(left.desc, right.desc, res.matches);
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    res.match_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    cv::Mat visRaw;
    cv::drawMatches(left.original, left.kpts, right.original, right.kpts,
                    res.matches, visRaw,
                    cv::Scalar(80,220,80), cv::Scalar(80,80,220),
                    std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    res.vis = letterbox(visRaw, slotSize);
    return res;
}


static std::string open_image_dialog(const char* title) {
    const char* filters[] = { "*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff" };
    const char* path = tinyfd_openFileDialog(
        title,
        /*defaultPathAndFile*/ "",
        /*numOfFilterPatterns*/ (int)(sizeof(filters)/sizeof(filters[0])),
        /*filterPatterns*/ filters,
        /*singleFilterDescription*/ "Image files",
        /*allowMultipleSelects*/ 0
    );
    return path ? std::string(path) : std::string();
}



// Utility: read path from stdin (non-blocking for UI because we pause the loop on purpose)
static std::string ask_path_in_console(const char* which) {
    std::string p;
    std::cout << "Enter " << which << " image path: " << std::flush;
    std::getline(std::cin, p);
    return p;
}





// ORB detection with timing. Returns kpts/desc and a visualization image.
static void detect_orb_and_render(const cv::Mat& imgBgr, int nfeatures, int fastThr,
                                  double scaleFactor, int nlevels,
                                  std::vector<cv::KeyPoint>& out_kpts, cv::Mat& out_desc,
                                  double& out_ms, cv::Mat& out_vis_for_slot, const cv::Size& slotSize)
{
    out_kpts.clear(); out_desc.release(); out_vis_for_slot.release();
    out_ms = 0.0;

    if (imgBgr.empty()) {
        out_vis_for_slot = letterbox(imgBgr, slotSize);
        return;
    }

    cv::Mat gray;
    if (imgBgr.channels() == 3) cv::cvtColor(imgBgr, gray, cv::COLOR_BGR2GRAY);
    else gray = imgBgr;

    // ORB: FAST corners + oriented BRIEF. Good default for binary descriptors.
    auto orb = cv::ORB::create(nfeatures, (float)scaleFactor, nlevels,
                               /*edgeThreshold*/31, /*firstLevel*/0,
                               /*WTA_K*/2, cv::ORB::HARRIS_SCORE, /*patchSize*/31, fastThr);

    auto t0 = std::chrono::high_resolution_clock::now();
    orb->detectAndCompute(gray, cv::noArray(), out_kpts, out_desc);
    auto t1 = std::chrono::high_resolution_clock::now();
    out_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Build a visualization: draw keypoints on a letterboxed copy
    cv::Mat preview = letterbox(imgBgr, slotSize);
    // We want to draw at the letterboxed scale so keypoint positions line up.
    // Draw on a temp that has same size as original, then letterbox? Simpler: scale kpts.
    double r = std::min(slotSize.width / (double)imgBgr.cols, slotSize.height / (double)imgBgr.rows);
    int xoff = (slotSize.width  - (int)std::round(imgBgr.cols * r)) / 2;
    int yoff = (slotSize.height - (int)std::round(imgBgr.rows * r)) / 2;
    cv::Mat draw = preview.clone();
    for (const auto& kp : out_kpts) {
        cv::Point2f p((float)(xoff + kp.pt.x * r), (float)(yoff + kp.pt.y * r));
        cv::circle(draw, p, 2, cv::Scalar(80,220,80), 1, cv::LINE_AA);
    }
    out_vis_for_slot = draw;
}


// cvui is header-only. We define CVUI_IMPLEMENTATION in exactly one .cpp:
#define CVUI_IMPLEMENTATION
#include "cvui.h"

namespace ui_defaults {
    // Canvas and panel sizes. Tweak if you prefer a different aspect.
    constexpr int CANVAS_W = 1600;
    constexpr int CANVAS_H = 900;
    constexpr int PANEL_W  = 340;   // right-side UI column
    constexpr int GAP      = 8;
}

// Simple state container for UI controls.
// We'll wire these into the actual pipeline in later steps.
struct AppState {
    // ORB params
    int    orb_nfeatures      = 1000;
    int    orb_fast_threshold = 20;
    double orb_scale_factor   = 1.2;  // [1.1, 1.6]
    int    orb_nlevels        = 8;

    // Matching (unused in this step)
    bool   use_flann          = false;
    double ratio_test         = 0.75;
    bool   cross_check        = false;

    // Essential / RANSAC (unused in this step)
    double ransac_thresh_px   = 1.0;
    double ransac_conf        = 0.999;
    bool   show_inliers_only  = true;

    bool   show_timing        = true;

    // Images and cached outputs for this step
    ImageSlot left, right;

    // Dirty flag to trigger recompute when params change
    int last_orb_nfeatures = orb_nfeatures;
    int last_orb_fast_threshold = orb_fast_threshold;
    double last_orb_scale_factor = orb_scale_factor;
    int last_orb_nlevels = orb_nlevels;

        // Matching cache
    MatchResult matches;
    int last_use_flann = -1;
    double last_ratio_test = -1.0;
    int last_cross_check = -1;

        // Intrinsics (optional)
    bool   use_intrinsics = false;
    double fx = 0, fy = 0, cx = 0, cy = 0;

    // Epipolar cache
    EpiResult epi;

    // Last-knowns for change detection (to avoid recompute every frame)
    double last_ransac_thresh = ransac_thresh_px;
    double last_ransac_conf   = ransac_conf;
    bool   last_use_intrinsics = use_intrinsics;
    double last_fx = fx, last_fy = fy, last_cx = cx, last_cy = cy;

        // In AppState:
    int  orb_version = 0;                   // bump when ORB recomputes
    int  last_orb_version_for_matches = -1; // last ORB used by matches
    int  match_version = 0;                 // bump when matches recompute
    int  last_match_version_for_epi = -1;   // last matches used by epipolar



};


// Helper to draw a labeled rectangle area (our three “slots”)
static void draw_slot(cv::Mat& canvas, const cv::Rect& roi, const std::string& title) {
    cv::rectangle(canvas, roi, cv::Scalar(70,70,70), 1, cv::LINE_AA);
    int baseLine = 0;
    const int pad = 6;
    cv::Size textSize = cv::getTextSize(title, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseLine);
    cv::Rect header(roi.x, roi.y, roi.width, textSize.height + 2*pad);
    cv::rectangle(canvas, header, cv::Scalar(45,45,45), cv::FILLED);
    cv::putText(canvas, title, {roi.x + pad, roi.y + textSize.height + pad - 2},
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(230,230,230), 1, cv::LINE_AA);

    // Use a named ROI (lvalue) so cvui::printf can take a cv::Mat&
    cv::Mat slot = canvas(roi);
    std::string hint = "Images not loaded yet.\nPress L (left) and R (right) to load in next step.";
    cvui::printf(slot, 10, header.height + 10, 0.5, 0xAAAAAA, hint.c_str());
}

// Lay out the three rows (features / matches / epipolar)
struct Layout {
    cv::Rect slot_features_left;
    cv::Rect slot_features_right;
    cv::Rect slot_matches_left;
    cv::Rect slot_matches_right;
    cv::Rect slot_epi_left;
    cv::Rect slot_epi_right;
    cv::Rect panel; // right-side UI

    static Layout compute() {
        using namespace ui_defaults;
        Layout L;

        const int grid_w = CANVAS_W - PANEL_W - GAP;
        const int col_w  = (grid_w - GAP) / 2;
        const int row_h  = (CANVAS_H - 4*GAP) / 3; // 3 rows (features, matches, epipolar)

        int x0 = GAP;
        int x1 = GAP + col_w + GAP;
        int y  = GAP;

        // Row 1: features
        L.slot_features_left  = {x0, y,  col_w, row_h};
        L.slot_features_right = {x1, y,  col_w, row_h};
        y += row_h + GAP;

        // Row 2: matches
        L.slot_matches_left   = {x0, y,  col_w, row_h};
        L.slot_matches_right  = {x1, y,  col_w, row_h};
        y += row_h + GAP;

        // Row 3: epipolar lines
        L.slot_epi_left       = {x0, y,  col_w, row_h};
        L.slot_epi_right      = {x1, y,  col_w, row_h};

        // Right-side panel
        L.panel = {grid_w + GAP, GAP, PANEL_W - GAP*2, CANVAS_H - GAP*2};
        return L;
    }
};

int main() {
    using namespace ui_defaults;

    const std::string WINDOW = "Epipolar Geometry Visualizer";
    cv::namedWindow(WINDOW, cv::WINDOW_AUTOSIZE);
    cv::Mat canvas(CANVAS_H, CANVAS_W, CV_8UC3);
    cvui::init(WINDOW);

    AppState state;
    auto layout = Layout::compute();

    // Main loop: just UI + placeholders for now
    while (true) {
        canvas.setTo(cv::Scalar(30,30,30)); // dark background

                // --- Draw the three result areas ---
        draw_slot(canvas, layout.slot_features_left,  "1) Features - Left");
        draw_slot(canvas, layout.slot_features_right, "1) Features - Right");
        draw_slot(canvas, layout.slot_matches_left,   "2) Matches - Left");
        draw_slot(canvas, layout.slot_matches_right,  "2) Matches - Right");
        draw_slot(canvas, layout.slot_epi_left,       "3) Epipolar - Left");
        draw_slot(canvas, layout.slot_epi_right,      "3) Epipolar - Right");


        // --- UI Panel drawn directly on the main canvas using absolute coords ---
        const int px = layout.panel.x;
        const int py = layout.panel.y;
        const int pw = layout.panel.width;
        const int ph = layout.panel.height;

        cvui::context(WINDOW); // ensure cvui targets our window

        cvui::window(canvas, px, py, pw, ph, "Controls");

        int y = 30;

        // Feature Detection (ORB)
        cvui::text(canvas, px + 10, py + y, "Feature Detection (ORB)"); y += 18;
        cvui::text(canvas, px + 10, py + y, "nfeatures");                y += 18;
        cvui::trackbar(canvas, px + 10, py + y, pw - 20, &state.orb_nfeatures, 100, 5000); y += 52;

        cvui::text(canvas, px + 10, py + y, "FAST threshold");           y += 18;
        cvui::trackbar(canvas, px + 10, py + y, pw - 20, &state.orb_fast_threshold, 5, 60); y += 52;

        cvui::text(canvas, px + 10, py + y, "Scale factor");             y += 18;
        cvui::trackbar(canvas, px + 10, py + y, pw - 20, &state.orb_scale_factor, 1.1, 1.6); y += 52;

        cvui::text(canvas, px + 10, py + y, "Pyramid levels");           y += 18;
        cvui::trackbar(canvas, px + 10, py + y, pw - 20, &state.orb_nlevels, 3, 12); y += 60;


        // --- Matching
        cvui::text(canvas, px + 10, py + y, "Matching");  y += 18;
        cvui::checkbox(canvas, px + 10, py + y, "Use FLANN (else BF Hamming)", &state.use_flann);
        y += 30;

        // Ratio test slider stays always available
        cvui::text(canvas, px + 10, py + y, "Ratio test"); y += 18;
        cvui::trackbar(canvas, px + 10, py + y, pw - 20, &state.ratio_test, 0.50, 0.95);
        y += 52;

        // Cross-check row: interactive when BF, visually disabled when FLANN
        if (state.use_flann) {
            // Make sure it’s off and render a disabled-looking row.
            state.cross_check = false;

            // (Optional) draw a dimmed bar to hint disabled
            cv::Rect dis(px + 8, py + y - 4, pw - 16, 28);
            cv::Mat roi = canvas(dis);
            cv::Mat overlay = roi.clone();
            overlay.setTo(cv::Scalar(40,40,40)); // darker
            cv::addWeighted(overlay, 0.5, roi, 0.5, 0.0, roi);

            cvui::text(canvas, px + 14, py + y + 16,
                    "Cross-check (BF only) [disabled for FLANN]");
            y += 36;

            cvui::text(canvas, px + 10, py + y, "(Cross-check not available with FLANN)");
            y += 18;
        } else {
            // Normal interactive checkbox (BF path)
            cvui::checkbox(canvas, px + 10, py + y, "Cross-check (BF only)", &state.cross_check);
            y += 36;

            if (state.cross_check) {
                cvui::text(canvas, px + 10, py + y, "(Ratio test ignored when cross-check is on)");
                y += 18;
            }
        }


        // Essential / RANSAC
        cvui::text(canvas, px + 10, py + y, "Essential / RANSAC");        y += 18;
        cvui::text(canvas, px + 10, py + y, "Inlier threshold (px)");     y += 18;
        cvui::trackbar(canvas, px + 10, py + y, pw - 20, &state.ransac_thresh_px, 0.2, 5.0); y += 52;

        cvui::text(canvas, px + 10, py + y, "Confidence");                y += 18;
        cvui::trackbar(canvas, px + 10, py + y, pw - 20, &state.ransac_conf, 0.90, 0.999); y += 52;

        cvui::checkbox(canvas, px + 10, py + y, "Show inliers only", &state.show_inliers_only); y += 36;
        cvui::checkbox(canvas, px + 10, py + y, "Show timing", &state.show_timing);             y += 30;

        // cvui::printf(canvas, px + 10, py + y, 0.5, 0xA0A0FF,
        //             "Next: add image loading + ORB.\n"
        //             "Hotkeys: L=Load Left, R=Load Right, Q=Quit");

                // --- Camera intrinsics (optional)
        cvui::text(canvas, px + 10, py + y, "Camera intrinsics (optional)"); y += 18;
        cvui::checkbox(canvas, px + 10, py + y, "Use intrinsics (Essential & pose)", &state.use_intrinsics);
        y += 30;

        cvui::text(canvas, px + 10, py + y, "fx"); y += 18;
        cvui::trackbar(canvas, px + 10, py + y, pw - 20, &state.fx, 100.0, 4000.0); y += 52;

        cvui::text(canvas, px + 10, py + y, "fy"); y += 18;
        cvui::trackbar(canvas, px + 10, py + y, pw - 20, &state.fy, 100.0, 4000.0); y += 52;

        cvui::text(canvas, px + 10, py + y, "cx"); y += 18;
        cvui::trackbar(canvas, px + 10, py + y, pw - 20, &state.cx, 0.0, 4000.0); y += 52;

        cvui::text(canvas, px + 10, py + y, "cy"); y += 18;
        cvui::trackbar(canvas, px + 10, py + y, pw - 20, &state.cy, 0.0, 4000.0); y += 52;


// ===== ORB detect (run if needed) =====
bool orbParamsChanged =
    state.orb_nfeatures != state.last_orb_nfeatures ||
    state.orb_fast_threshold != state.last_orb_fast_threshold ||
    std::abs(state.orb_scale_factor - state.last_orb_scale_factor) > 1e-9 ||
    state.orb_nlevels != state.last_orb_nlevels;

// Update “last” and decide per-side recompute
if (orbParamsChanged) {
    state.last_orb_nfeatures      = state.orb_nfeatures;
    state.last_orb_fast_threshold = state.orb_fast_threshold;
    state.last_orb_scale_factor   = state.orb_scale_factor;
    state.last_orb_nlevels        = state.orb_nlevels;
    // Clear per-side vis to signal recompute
    if (state.left.valid())  state.left.vis.release();
    if (state.right.valid()) state.right.vis.release();
}

bool recomputed_orb = false;

// Left
if (state.left.valid() && state.left.vis.empty()) {
    detect_orb_and_render(
        state.left.original,
        state.orb_nfeatures, state.orb_fast_threshold,
        state.orb_scale_factor, state.orb_nlevels,
        state.left.kpts, state.left.desc, state.left.detect_ms, state.left.vis,
        cv::Size(layout.slot_features_left.width, layout.slot_features_left.height)
    );
    recomputed_orb = true;
}
// Right
if (state.right.valid() && state.right.vis.empty()) {
    detect_orb_and_render(
        state.right.original,
        state.orb_nfeatures, state.orb_fast_threshold,
        state.orb_scale_factor, state.orb_nlevels,
        state.right.kpts, state.right.desc, state.right.detect_ms, state.right.vis,
        cv::Size(layout.slot_features_right.width, layout.slot_features_right.height)
    );
    recomputed_orb = true;
}

if (recomputed_orb) {
    state.orb_version++;                        // upstream changed
    state.last_orb_version_for_matches = -1;    // force matches next
    state.last_match_version_for_epi   = -1;    // and epi after that
}

// Blit features row
if (state.left.valid()) {
    cv::Mat slotL = canvas(layout.slot_features_left);
    state.left.vis.copyTo(slotL);
    put_info(slotL, 10, 22, "path: " + state.left.path);
    put_info(slotL, 10, 44, "features: " + std::to_string(state.left.kpts.size()));
    if (state.show_timing) put_info(slotL, 10, 66, "detect: " + std::to_string((int)std::round(state.left.detect_ms)) + " ms");
}
if (state.right.valid()) {
    cv::Mat slotR = canvas(layout.slot_features_right);
    state.right.vis.copyTo(slotR);
    put_info(slotR, 10, 22, "path: " + state.right.path);
    put_info(slotR, 10, 44, "features: " + std::to_string(state.right.kpts.size()));
    if (state.show_timing) put_info(slotR, 10, 66, "detect: " + std::to_string((int)std::round(state.right.detect_ms)) + " ms");
}

// ===== Matches (run if needed) =====
bool matchParamsChanged =
    (state.use_flann != (bool)state.last_use_flann) ||
    (std::abs(state.ratio_test - state.last_ratio_test) > 1e-9) ||
    ((int)state.cross_check != state.last_cross_check);

bool have_desc = state.left.valid() && state.right.valid() &&
                 !state.left.desc.empty() && !state.right.desc.empty();

bool needMatches =
    have_desc &&
    (matchParamsChanged ||
     state.last_orb_version_for_matches != state.orb_version ||
     state.matches.matches.empty());

if (needMatches) {
    state.last_use_flann   = state.use_flann;
    state.last_ratio_test  = state.ratio_test;
    state.last_cross_check = state.cross_check;

    state.matches = match_features(
        state.left, state.right,
        state.use_flann, state.ratio_test, state.cross_check,
        cv::Size(layout.slot_matches_left.width * 2 + ui_defaults::GAP,
                 layout.slot_matches_left.height)
    );

    state.last_orb_version_for_matches = state.orb_version;
    state.match_version++;
    state.last_match_version_for_epi = -1; // force epi next
}

// Blit matches row
if (state.left.valid() && state.right.valid() && !state.matches.vis.empty()) {
    cv::Mat vis = state.matches.vis;
    int mid = vis.cols / 2;
    cv::Rect leftROI(0, 0, mid, vis.rows);
    cv::Rect rightROI(mid, 0, vis.cols - mid, vis.rows);

    cv::resize(vis(leftROI),  canvas(layout.slot_matches_left),  layout.slot_matches_left.size());
    cv::resize(vis(rightROI), canvas(layout.slot_matches_right), layout.slot_matches_right.size());

    cv::Mat slotML = canvas(layout.slot_matches_left);
    put_info(slotML, 10, 22, "Matches: " + std::to_string(state.matches.matches.size()));
    if (state.show_timing) {
        put_info(slotML, 10, 44, "time: " + std::to_string((int)std::round(state.matches.match_ms)) + " ms");
    }
}

// ===== Epipolar (run if needed) =====
bool epiParamsChanged =
    state.ransac_thresh_px != state.last_ransac_thresh ||
    std::abs(state.ransac_conf - state.last_ransac_conf) > 1e-12 ||
    state.use_intrinsics != state.last_use_intrinsics ||
    std::abs(state.fx - state.last_fx) > 1e-9 ||
    std::abs(state.fy - state.last_fy) > 1e-9 ||
    std::abs(state.cx - state.last_cx) > 1e-9 ||
    std::abs(state.cy - state.last_cy) > 1e-9;

bool have_matches = state.matches.matches.size() >= 8;

bool needEpi =
    have_matches &&
    (epiParamsChanged ||
     state.last_match_version_for_epi != state.match_version ||
     state.epi.visLeft.empty());

if (needEpi) {
    state.last_ransac_thresh  = state.ransac_thresh_px;
    state.last_ransac_conf    = state.ransac_conf;
    state.last_use_intrinsics = state.use_intrinsics;
    state.last_fx = state.fx; state.last_fy = state.fy;
    state.last_cx = state.cx; state.last_cy = state.cy;

    state.epi = compute_epi_and_render(
        state.left, state.right, state.matches.matches,
        state.use_intrinsics, state.fx, state.fy, state.cx, state.cy,
        state.ransac_thresh_px, state.ransac_conf,
        cv::Size(layout.slot_epi_left.width, layout.slot_epi_left.height),
        cv::Size(layout.slot_epi_right.width, layout.slot_epi_right.height)
    );
    state.last_match_version_for_epi = state.match_version;
}

// Blit epipolar row (show something even if not enough matches)
if (!state.epi.visLeft.empty()) {
    state.epi.visLeft.copyTo(canvas(layout.slot_epi_left));
    state.epi.visRight.copyTo(canvas(layout.slot_epi_right));

    cv::Mat slotEL = canvas(layout.slot_epi_left);
    int inl = state.epi.inliers, tot = state.epi.total;
    double pct = (tot > 0) ? (100.0 * inl / tot) : 0.0;
    put_info(slotEL, 10, 22,
             (state.epi.usedEssential ? "E (with K)" : "F (no K)")
             + std::string(" | inliers: ") + std::to_string(inl) + "/" + std::to_string(tot)
             + std::string(" (") + std::to_string((int)std::round(pct)) + "%)");

    if (state.show_timing) {
        put_info(slotEL, 10, 44, "RANSAC: " + std::to_string((int)std::round(state.epi.ransac_ms)) + " ms");
    }
} else {
    // Fallback: show plain letterboxed images so row 3 is never blank
    if (state.left.valid())  letterbox(state.left.original,  layout.slot_epi_left.size())
                                .copyTo(canvas(layout.slot_epi_left));
    if (state.right.valid()) letterbox(state.right.original, layout.slot_epi_right.size())
                                .copyTo(canvas(layout.slot_epi_right));
}






        // keep these after all widgets:
        cvui::update();
        cvui::imshow(WINDOW, canvas);
        int key = cv::waitKey(20);
                // Handle hotkeys: L = load left, R = load right
        if (key == 'l' || key == 'L') {
            std::string p = open_image_dialog("Select LEFT image");
            if (!p.empty()) {
                state.left = ImageSlot{};
                state.left.path = p;
                state.left.original = cv::imread(p, cv::IMREAD_COLOR);
                if (state.left.original.empty()) {
                    tinyfd_messageBox("Load error", ("Failed to load LEFT image:\n" + p).c_str(), "ok", "error", 1);
                }
            }
                        // After loading LEFT image successfully
            if (state.left.valid()) {
                if (state.fx <= 0 || state.fy <= 0) {
                    int w = state.left.original.cols, h = state.left.original.rows;
                    state.fx = state.fy = (double)std::max(w, h);
                    state.cx = w / 2.0;
                    state.cy = h / 2.0;
                }
                            // Clear downstream so we recompute immediately
            state.left.vis.release();   // or state.right.vis.release();
            state.matches = MatchResult{};
            state.epi     = EpiResult{};
            state.last_orb_version_for_matches = -1;
            state.last_match_version_for_epi   = -1;

}

        }

        if (key == 'r' || key == 'R') {
            std::string p = open_image_dialog("Select RIGHT image");
            if (!p.empty()) {
                state.right = ImageSlot{};
                state.right.path = p;
                state.right.original = cv::imread(p, cv::IMREAD_COLOR);
                if (state.right.original.empty()) {
                    tinyfd_messageBox("Load error", ("Failed to load RIGHT image:\n" + p).c_str(), "ok", "error", 1);
                }
            }
                            // Clear downstream so we recompute immediately
                state.right.vis.release();   // or state.right.vis.release();
                state.matches = MatchResult{};
                state.epi     = EpiResult{};
                state.last_orb_version_for_matches = -1;
                state.last_match_version_for_epi   = -1;

        }


        if (key == 'q' || key == 27) break;


    }

    return 0;
}
