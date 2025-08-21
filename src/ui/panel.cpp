#include "ui/panel.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <algorithm>
#include <sstream>
#include <filesystem>

#if __has_include(<opencv2/freetype.hpp>)
  #include <opencv2/freetype.hpp>
  #define HAS_FT 1
#else
  #define HAS_FT 0
#endif

using std::string;
namespace fs = std::filesystem;

// =================== global metrics / zoom ===================
static double g_fontScaleMul = 1.0; // user zoom
static int    g_lineH_base   = 22;  // base info text line height
static int    g_header_base  = 28;  // base header height
static int    g_controlsW0   = 520; // base min sizes
static int    g_controlsH0   = 300;
static int    g_infoW0       = 520;
static int    g_infoH0       = 360;

static inline int SZ(double x) { return (int)std::round(x * g_fontScaleMul); }

void uiSetGlobalFontScale(double mul, UIState* uiOpt) {
    g_fontScaleMul = std::clamp(mul, 0.6, 2.0);
    if (uiOpt) {
        auto clampRect = [&](const cv::Rect& r, const cv::Size& canvas) {
            int x = std::clamp(r.x, 0, std::max(0, canvas.width  - r.width));
            int y = std::clamp(r.y, 0, std::max(0, canvas.height - r.height));
            return cv::Rect{x,y,r.width,r.height};
        };
        auto ensureMin = [&](cv::Rect& r, int w0, int h0){
            int minW = SZ(w0), minH = SZ(h0);
            if (r.width  < minW) r.width  = minW;
            if (r.height < minH) r.height = minH;
        };
        ensureMin(uiOpt->controlsPanel.rect, g_controlsW0, g_controlsH0);
        ensureMin(uiOpt->infoPanel.rect,     g_infoW0,     g_infoH0);
        uiOpt->controlsPanel.rect = clampRect(uiOpt->controlsPanel.rect, uiOpt->canvasSize);
        uiOpt->infoPanel.rect     = clampRect(uiOpt->infoPanel.rect,     uiOpt->canvasSize);
        uiOpt->controlsPanel.headerH = SZ(g_header_base);
        uiOpt->infoPanel.headerH     = SZ(g_header_base);
        // Reflow layout on zoom, but preserve dragged positions

    }
    if (uiOpt && uiOpt->docked) {
    uiUpdateDockBounds(*uiOpt, uiOpt->dockX, uiOpt->dockW, uiOpt->dockH);
}
}
double uiGetGlobalFontScale() { return g_fontScaleMul; }

// =================== FreeType-backed solid text ===================
namespace {
#if HAS_FT
cv::Ptr<cv::freetype::FreeType2> g_ft;
#endif
bool        g_ft_ok = false;
std::string g_font_path;

static bool tryInitFT() {
#if HAS_FT
    if (g_ft_ok) return true;
    const char* cands[] = {
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf"
    };
    for (auto* p : cands) if (fs::exists(p)) { g_font_path = p; break; }
    if (g_font_path.empty()) return false;
    try { g_ft = cv::freetype::createFreeType2(); g_ft->loadFontData(g_font_path, 0); g_ft_ok=true; }
    catch (...) { g_ft_ok=false; }
#endif
    return g_ft_ok;
}

// solid white by default; bold-ish if called twice by the caller
static void drawText(cv::Mat& img, const std::string& s, cv::Point org,
                     double scale, cv::Scalar col, int thick=1) {
    scale *= g_fontScaleMul;
    if (tryInitFT()) {
#if HAS_FT
        int px = (int)std::round(20.0 * scale);
        g_ft->putText(img, s, org, px, col, -1 /*filled*/, cv::LINE_AA, true);
        return;
#endif
    }
    // Hershey fallback (stroke only) — use larger thickness for legibility
    cv::putText(img, s, org, cv::FONT_HERSHEY_SIMPLEX, scale,
                col, std::max(2, thick), cv::LINE_AA);
}
} // namespace

// =================== helpers ===================
static cv::Rect clampRect(const cv::Rect& r, const cv::Size& canvas) {
    int x = std::clamp(r.x, 0, std::max(0, canvas.width  - r.width));
    int y = std::clamp(r.y, 0, std::max(0, canvas.height - r.height));
    return {x,y,r.width,r.height};
}

static void ensureMinPanelSize(Panel& P, int w0, int h0) {
    P.headerH = SZ(g_header_base);
    int minW = SZ(w0), minH = SZ(h0);
    if (P.rect.width  < minW) P.rect.width  = minW;
    if (P.rect.height < minH) P.rect.height = minH;
}

static std::vector<string> wrapToWidth(const std::vector<string>& lines, int maxW,
                                       double scale=0.55, int thick=1) {
    scale *= g_fontScaleMul;
    std::vector<string> out;
    for (const auto& ln : lines) {
        std::istringstream is(ln);
        string cur, w;
        while (is >> w) {
            string t = cur.empty() ? w : cur + " " + w;
            int base=0; cv::Size ts = cv::getTextSize(t, cv::FONT_HERSHEY_SIMPLEX, scale, thick, &base);
            if (ts.width <= maxW) cur = std::move(t);
            else { if (!cur.empty()) out.push_back(cur); cur = w; }
        }
        if (!cur.empty()) out.push_back(cur);
    }
    return out;
}

// =================== API impl ===================
void uiInit(UIState& ui, const cv::Size& canvasSize) {
    ui.canvasSize = canvasSize;
    ui.docked = true;
    ui.controlsPanel = {"Controls", {12,12,  g_controlsW0, g_controlsH0}};
    ui.infoPanel     = {"Learn",    {12,320, g_infoW0,     g_infoH0}};
    ui.infoPanel.scrollable = true;
    uiSetGlobalFontScale(uiGetGlobalFontScale(), &ui);
}

void uiDockLeft(UIState& ui, int x, int width, int height) {
    ui.dockX = x; ui.dockW = width; ui.dockH = height; ui.docked = true;
    const int pad = SZ(12);
    ui.controlsPanel.rect = { x+pad, pad, width-2*pad, std::max(SZ(280), ui.controlsPanel.rect.height) };
    int infoY = ui.controlsPanel.rect.y + ui.controlsPanel.rect.height + pad;
    ui.infoPanel.rect = { x+pad, infoY, width-2*pad, std::max(SZ(220), height - infoY - pad) };
    ui.controlsPanel.headerH = SZ(28);
    ui.infoPanel.headerH     = SZ(28);
}

void uiUpdateDockBounds(UIState& ui, int x, int width, int height) {
    // Update dock geometry and keep panels draggable inside it
    ui.dockX = x; ui.dockW = width; ui.dockH = height; ui.docked = true;

    int pad = SZ(12);
    auto clampToDock = [&](cv::Rect r){
        // ensure panel has current header height
        // (width/height will be adjusted below)
        int minX = ui.dockX + pad;
        int maxX = ui.dockX + ui.dockW - r.width - pad;
        int minY = pad;
        int maxY = ui.dockH - r.height - pad;
        r.x = std::clamp(r.x, minX, std::max(minX, maxX));
        r.y = std::clamp(r.y, minY, std::max(minY, maxY));
        return r;
    };

    // Keep current Y (user can drag), but update widths to fill dock
    int newW = std::max(SZ(260), width - 2*pad);
    ui.controlsPanel.headerH = SZ(28);
    ui.infoPanel.headerH     = SZ(28);

    // If first time (0,0), initialize a sensible layout
    if (ui.controlsPanel.rect.x == 0 && ui.controlsPanel.rect.y == 0) {
        ui.controlsPanel.rect = { x + pad, pad, newW, std::max(SZ(280), ui.controlsPanel.rect.height) };
    } else {
        ui.controlsPanel.rect.width = newW;
    }

    if (ui.infoPanel.rect.x == 0 && ui.infoPanel.rect.y == 0) {
        int infoY = ui.controlsPanel.rect.y + ui.controlsPanel.rect.height + pad;
        int infoH = std::max(SZ(220), height - infoY - pad);
        ui.infoPanel.rect = { x + pad, infoY, newW, infoH };
    } else {
        ui.infoPanel.rect.width = newW;
        // If the dock got shorter (e.g., window resized), keep the current height but clamp to fit
        if (ui.infoPanel.rect.y + ui.infoPanel.rect.height + pad > height) {
            ui.infoPanel.rect.height = std::max(SZ(180), height - ui.infoPanel.rect.y - pad);
        }
    }

    // Clamp final positions into the dock
    ui.controlsPanel.rect = clampToDock(ui.controlsPanel.rect);
    ui.infoPanel.rect     = clampToDock(ui.infoPanel.rect);
}

static void drawHeader(cv::Mat& canvas, Panel& p) {
    cv::rectangle(canvas, p.rect, {30,30,30}, cv::FILLED, cv::LINE_AA);
    cv::rectangle(canvas, p.rect, {80,80,80}, 1, cv::LINE_AA);
    cv::rectangle(canvas, {p.rect.x, p.rect.y, p.rect.width, p.headerH},
                  {45,45,45}, cv::FILLED, cv::LINE_AA);
    drawText(canvas, p.title, {p.rect.x+SZ(10), p.rect.y + p.headerH - SZ(6)}, 0.55, {255,255,255}, 1);
}

void uiDrawControls(cv::Mat& canvas, UIState& ui, const std::string& header) {
    auto& P = ui.controlsPanel;
    ensureMinPanelSize(P, g_controlsW0, g_controlsH0);
    drawHeader(canvas, P);

    const int x0 = P.rect.x + SZ(10);
    int y = P.rect.y + P.headerH + SZ(22);

    const int lineH = SZ(32);
    const int nameW = SZ(180);
    const int valW  = SZ(90);
    const int barW  = P.rect.width - SZ(20) - nameW - valW;
    const int barH  = std::max(SZ(12), 10);

    // Section header
    drawText(canvas, header, {x0, y}, 0.62, {255,255,255}, 1);
    y += SZ(10);

    // Metrics strip (colored, bold-ish)
    if (!ui.metrics.empty()) {
        int lineH_metric = SZ(26);
        for (const auto& m : ui.metrics) {
            drawText(canvas, m.label, {x0, y + SZ(13)}, 0.50, {255,255,255}, 1);
            drawText(canvas, m.value, {x0 + SZ(140), y + SZ(13)}, m.scale, m.color, 1);
            drawText(canvas, m.value, {x0 + SZ(141), y + SZ(13)}, m.scale, m.color, 1); // faux bold
            y += lineH_metric;
        }
        y += SZ(6);
    }

    // Sliders
    for (size_t i=0;i<ui.sliders.size();++i) {
        auto& s = ui.sliders[i];
        y += lineH;

        s.nameRect = {x0, y-SZ(18), nameW, SZ(18)};
        s.barRect  = {x0 + nameW,   y-SZ(20), barW, barH};
        s.valRect  = {s.barRect.x + s.barRect.width + SZ(6), y-SZ(18), valW-SZ(6), SZ(18)};

        drawText(canvas, s.name, {s.nameRect.x+SZ(6), s.nameRect.y+SZ(13)}, 0.50, {255,255,255}, 1);
        cv::rectangle(canvas, s.barRect, {60,60,60}, cv::FILLED, cv::LINE_AA);
        cv::rectangle(canvas, s.barRect, {120,120,120}, 1, cv::LINE_AA);

        float t = (s.val - s.minv) / std::max(1e-9f, (s.maxv - s.minv));
        int kx = s.barRect.x + std::clamp((int)std::lround(t*s.barRect.width), 0, s.barRect.width);
        cv::Rect knob(kx-SZ(4), s.barRect.y-SZ(3), SZ(8), s.barRect.height+SZ(6));
        cv::rectangle(canvas, knob, {180,180,180}, cv::FILLED, cv::LINE_AA);
        cv::rectangle(canvas, knob, {40,40,40}, 1, cv::LINE_AA);

        char buf[64];
        if (s.isInt) std::snprintf(buf, sizeof(buf), "%d", (int)std::lround(s.val));
        else         std::snprintf(buf, sizeof(buf), "%.4f", s.val);
        drawText(canvas, buf, {s.valRect.x + SZ(4), s.valRect.y + SZ(13)}, 0.50, {220,220,220}, 1);
    }

    // Checkboxes
    for (size_t i=0;i<ui.checks.size();++i) {
        auto& c = ui.checks[i];
        y += lineH;
        c.boxRect  = {x0,         y-SZ(18), SZ(18), SZ(18)};
        c.nameRect = {x0+SZ(26),  y-SZ(18), P.rect.width - SZ(36), SZ(18)};

        cv::rectangle(canvas, c.boxRect, {230,230,230}, 1, cv::LINE_AA);
        if (c.checked) {
            cv::rectangle(canvas, c.boxRect, {90,200,90}, cv::FILLED, cv::LINE_AA);
            cv::line(canvas, c.boxRect.tl()+cv::Point(SZ(3),SZ(9)), c.boxRect.tl()+cv::Point(SZ(8),SZ(14)), {20,60,20}, 2, cv::LINE_AA);
            cv::line(canvas, c.boxRect.tl()+cv::Point(SZ(8),SZ(14)), c.boxRect.tl()+cv::Point(SZ(15),SZ(4)), {20,60,20}, 2, cv::LINE_AA);
        }
        drawText(canvas, c.name, {c.nameRect.x, c.nameRect.y+SZ(13)}, 0.50, {255,255,255}, 1);
    }

    // Buttons row
    int bx = x0;
    int by = P.rect.y + P.rect.height - SZ(34);
    for (size_t i=0;i<ui.buttons.size(); ++i) {
        auto& b = ui.buttons[i];
        int base=0; auto ts = cv::getTextSize(b.label, cv::FONT_HERSHEY_SIMPLEX, 0.5*g_fontScaleMul, 1, &base);
        int bw = ts.width + SZ(22), bh = SZ(26);
        b.rect = {bx, by, bw, bh};
        cv::Scalar bg = b.primary ? cv::Scalar(60,120,230) : cv::Scalar(70,70,70);
        cv::rectangle(canvas, b.rect, bg, cv::FILLED, cv::LINE_AA);
        cv::rectangle(canvas, b.rect, {200,200,200}, 1, cv::LINE_AA);
        drawText(canvas, b.label, {b.rect.x + SZ(8), b.rect.y + bh - SZ(8)}, 0.50, {255,255,255}, 1);
        bx += bw + SZ(8);
    }

    // Tooltips (hovering over parameter names only)
    if (ui.hoverType != -1 && ui.hoverIndex != -1) {
        std::string tip = (ui.hoverType==0) ? ui.sliders[ui.hoverIndex].tip
                                            : ui.checks[ui.hoverIndex].tip;
        int base=0; auto ts = cv::getTextSize(tip, cv::FONT_HERSHEY_SIMPLEX, 0.5*g_fontScaleMul, 1, &base);
        int tx = std::min(std::max(SZ(12), ui.mouse.x+SZ(14)), canvas.cols - ts.width - SZ(12));
        int ty = std::min(std::max(SZ(40), ui.mouse.y+SZ(14)), canvas.rows - ts.height - SZ(12));
        cv::Rect box(tx-SZ(8), ty-SZ(8), ts.width+SZ(16), ts.height+SZ(16));
        cv::rectangle(canvas, box, {30,30,30}, cv::FILLED, cv::LINE_AA);
        cv::rectangle(canvas, box, {200,200,200}, 1, cv::LINE_AA);
        drawText(canvas, tip, {tx, ty + ts.height/2}, 0.50, {255,255,255}, 1);
    }
}

void uiDrawInfo(cv::Mat& canvas, UIState& ui,
                const std::string& title,
                const std::vector<std::string>& linesIn) {
    auto& P = ui.infoPanel;
    ensureMinPanelSize(P, g_infoW0, g_infoH0);
    drawHeader(canvas, P);

    cv::Rect body{P.rect.x+SZ(6), P.rect.y + P.headerH + SZ(4),
                  P.rect.width-SZ(12), P.rect.height - P.headerH - SZ(8)};
    cv::rectangle(canvas, body, {35,35,35}, cv::FILLED, cv::LINE_AA);

    // wrap, clamp, and store for mouse handling
    auto wrapped = wrapToWidth(linesIn, body.width - SZ(16));
    int lineH = SZ(g_lineH_base);
    int maxLines = std::max(1, (body.height - SZ(20)) / lineH);
    P.scrollOffset = std::clamp(P.scrollOffset, 0, std::max(0, (int)wrapped.size()-maxLines));

    ui.infoBodyRect          = body;
    ui.infoWrappedTotalLines = (int)wrapped.size();
    ui.infoMaxVisibleLines   = maxLines;
    ui.infoLineH             = lineH;

    // draw text
    int x = body.x + SZ(10), y = body.y + SZ(16);
    for (int i=0;i<maxLines && P.scrollOffset+i < (int)wrapped.size(); ++i) {
        drawText(canvas, wrapped[P.scrollOffset+i], {x, y}, 0.55, {255,255,255}, 1);
        y += lineH;
    }

    // scrollbar (+ store rects)
    float frac = std::min(1.f, std::max(0.f, (float)maxLines / std::max(1, (int)wrapped.size())));
    int barH = std::max(SZ(18), (int)(frac * (body.height-SZ(12))));
    int trackH = body.height - SZ(12);
    int barY = body.y + SZ(6) + (int)((float)P.scrollOffset /
             std::max(1, (int)wrapped.size()-maxLines) * (trackH - barH));
    ui.infoScrollTrack = {body.x + body.width - SZ(8), body.y + SZ(6), SZ(6), trackH};
    ui.infoScrollThumb = {ui.infoScrollTrack.x, barY, ui.infoScrollTrack.width, barH};
    cv::rectangle(canvas, ui.infoScrollTrack, {70,70,70}, cv::FILLED, cv::LINE_AA);
    cv::rectangle(canvas, ui.infoScrollThumb, {160,160,160}, cv::FILLED, cv::LINE_AA);
}

void uiHandleMouse(int event, int x, int y, int flags, void* userdata) {
    auto* ui = reinterpret_cast<UIState*>(userdata);
    ui->mouse = {x,y};

    auto headerRect = [](const Panel& P){ return cv::Rect(P.rect.x, P.rect.y, P.rect.width, P.headerH); };

    // Hover for tooltips on parameter names
    ui->hoverType=-1; ui->hoverIndex=-1;
    for (int i=0;i<(int)ui->sliders.size();++i)
        if (ui->sliders[i].nameRect.contains(ui->mouse)) { ui->hoverType=0; ui->hoverIndex=i; break; }
    if (ui->hoverType==-1)
        for (int i=0;i<(int)ui->checks.size();++i)
            if (ui->checks[i].nameRect.contains(ui->mouse)) { ui->hoverType=1; ui->hoverIndex=i; break; }

    if (event == cv::EVENT_LBUTTONDOWN) {
        // Start dragging if cursor is on a panel header (always allowed)
        if (headerRect(ui->controlsPanel).contains(ui->mouse)) {
            ui->controlsPanel.dragging = true;
            ui->controlsPanel.dragOff = ui->mouse - ui->controlsPanel.rect.tl();
        } else if (headerRect(ui->infoPanel).contains(ui->mouse)) {
            ui->infoPanel.dragging = true;
            ui->infoPanel.dragOff = ui->mouse - ui->infoPanel.rect.tl();
        }

        // sliders
        for (int i=0;i<(int)ui->sliders.size();++i)
            if (ui->sliders[i].barRect.contains(ui->mouse)) { ui->draggingSlider=i; break; }

        // checkboxes
        for (auto& c : ui->checks)
            if (c.boxRect.contains(ui->mouse) || c.nameRect.contains(ui->mouse)) c.checked = !c.checked;

        // buttons
        ui->clickedButton = -1;
        for (int i=0;i<(int)ui->buttons.size();++i)
            if (ui->buttons[i].rect.contains(ui->mouse)) { ui->clickedButton = i; break; }

        // Learn-panel scrollbar
        if (ui->infoScrollThumb.contains(ui->mouse)) {
            ui->draggingInfoScroll = true;
            ui->dragStartY = y;
            ui->dragStartOffset = ui->infoPanel.scrollOffset;
        } else if (ui->infoScrollTrack.contains(ui->mouse) && !ui->infoScrollThumb.contains(ui->mouse)) {
            int dir = (y < ui->infoScrollThumb.y) ? -1 : +1;
            int total = std::max(0, ui->infoWrappedTotalLines - ui->infoMaxVisibleLines);
            ui->infoPanel.scrollOffset = std::clamp(ui->infoPanel.scrollOffset + dir*ui->infoMaxVisibleLines, 0, total);
        }
    }
    else if (event == cv::EVENT_MOUSEMOVE) {
        auto clampToCanvas = [&](cv::Rect r){
            return cv::Rect(
                std::clamp(r.x, 0, std::max(0, ui->canvasSize.width  - r.width)),
                std::clamp(r.y, 0, std::max(0, ui->canvasSize.height - r.height)),
                r.width, r.height);
        };
        auto clampToDock = [&](cv::Rect r){
            int pad = SZ(12);
            int minX = ui->dockX + pad;
            int maxX = ui->dockX + ui->dockW - r.width - pad;
            int minY = pad;
            int maxY = ui->dockH - r.height - pad;
            r.x = std::clamp(r.x, minX, std::max(minX, maxX));
            r.y = std::clamp(r.y, minY, std::max(minY, maxY));
            return r;
        };

        if (ui->controlsPanel.dragging) {
            cv::Rect r = ui->controlsPanel.rect;
            r.x = x - ui->controlsPanel.dragOff.x;
            r.y = y - ui->controlsPanel.dragOff.y;
            ui->controlsPanel.rect = ui->docked ? clampToDock(r) : clampToCanvas(r);
        }
        if (ui->infoPanel.dragging) {
            cv::Rect r = ui->infoPanel.rect;
            r.x = x - ui->infoPanel.dragOff.x;
            r.y = y - ui->infoPanel.dragOff.y;
            ui->infoPanel.rect = ui->docked ? clampToDock(r) : clampToCanvas(r);
        }

        // slider drag
        if (ui->draggingSlider != -1) {
            auto& s = ui->sliders[ui->draggingSlider];
            float t = (x - s.barRect.x) / (float)s.barRect.width;
            t = std::clamp(t, 0.f, 1.f);
            s.val = s.minv + t*(s.maxv - s.minv);
            if (s.isInt) s.val = std::round(s.val);
        }

        // Learn scrollbar thumb drag
        if (ui->draggingInfoScroll) {
            int dy = y - ui->dragStartY;
            int trackSpan = std::max(1, ui->infoScrollTrack.height - ui->infoScrollThumb.height);
            float frac = std::clamp((float)dy / (float)trackSpan, -1.f, 1.f);
            int total = std::max(0, ui->infoWrappedTotalLines - ui->infoMaxVisibleLines);
            int deltaLines = (int)std::round(frac * total);
            ui->infoPanel.scrollOffset = std::clamp(ui->dragStartOffset + deltaLines, 0, total);
        }
    }
    else if (event == cv::EVENT_LBUTTONUP) {
        ui->controlsPanel.dragging = false;
        ui->infoPanel.dragging     = false;
        ui->draggingSlider         = -1;
        ui->draggingInfoScroll     = false;
    }
    else if (event == cv::EVENT_MOUSEWHEEL) {
        if (ui->infoBodyRect.contains(ui->mouse)) {
            int delta = cv::getMouseWheelDelta(flags);
            int step = (delta > 0 ? -3 : +3);
            int total = std::max(0, ui->infoWrappedTotalLines - ui->infoMaxVisibleLines);
            ui->infoPanel.scrollOffset = std::clamp(ui->infoPanel.scrollOffset + step, 0, total);
        }
    }
}
