#pragma once
#include <opencv2/core.hpp>
#include <string>
#include <vector>

// =================== UI primitives ===================
struct Slider {
    std::string name, tip;
    float minv=0.f, maxv=1.f, val=0.f;
    bool  isInt=false;
    cv::Rect nameRect, barRect, valRect;   // filled at draw time
};

struct Checkbox {
    std::string name, tip;
    bool checked=false;
    cv::Rect boxRect, nameRect;            // filled at draw time
};

struct Button {
    std::string label;
    bool primary=false;
    cv::Rect rect;                          // filled at draw time
};

// colored metric line (shown at top of Controls)
struct Metric {
    std::string label, value;
    cv::Scalar  color{60,220,60};           // value color
    double      scale = 0.65;               // value font scale (labels use 0.50)
};

struct Panel {
    std::string title;
    cv::Rect  rect;          // panel box
    int       headerH = 28;  // draggable header height
    bool      dragging=false;
    cv::Point dragOff{0,0};
    bool      scrollable=false;
    int       scrollOffset=0; // for info panel (in wrapped line units)
};

struct UIState {
    // Layout: docked left column
    bool docked = true;
    int  dockX = 0, dockW = 520, dockH = 600;

    // Panels
    Panel controlsPanel;
    Panel infoPanel;

    // Widgets
    std::vector<Slider>   sliders;
    std::vector<Checkbox> checks;
    std::vector<Button>   buttons;
    std::vector<Metric>   metrics;

    // Hover / interaction
    int hoverType=-1, hoverIndex=-1; // 0 slider-name, 1 check-name
    int draggingSlider=-1;
    int clickedButton=-1;

    // Reliable scroll geometry (set each frame by renderer)
    cv::Rect infoBodyRect, infoScrollTrack, infoScrollThumb;
    int infoWrappedTotalLines=0, infoMaxVisibleLines=0, infoLineH=22;
    bool draggingInfoScroll=false;
    int  dragStartY=0, dragStartOffset=0;

    // Canvas
    cv::Size  canvasSize{0,0};
    cv::Point mouse{0,0};
};

// =================== API ===================
void uiInit(UIState& ui, const cv::Size& canvasSize);
void uiDockLeft(UIState& ui, int x, int width, int height); // dock sidebar

// Draw panels (call every frame after populating ui.sliders/checks/buttons/metrics)
void uiDrawControls(cv::Mat& canvas, UIState& ui, const std::string& header);
void uiDrawInfo(cv::Mat& canvas, UIState& ui,
                const std::string& title,
                const std::vector<std::string>& lines);

// Mouse handling (cv::setMouseCallback(window, uiHandleMouse, &ui))
void uiHandleMouse(int event, int x, int y, int flags, void* userdata);

// Global font & layout zoom (affects text size, spacing, and panel min sizes)
void   uiSetGlobalFontScale(double mul, UIState* uiOpt=nullptr); // 0.6..2.0
double uiGetGlobalFontScale();
void uiUpdateDockBounds(UIState& ui, int x, int width, int height);

