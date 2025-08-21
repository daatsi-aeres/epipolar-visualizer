#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>

#include "app/intrinsics.h"
#include "teach/teach.h"

namespace fs = std::filesystem;

static bool hasFlag(int argc, char** argv, const std::string& f) {
    for (int i = 0; i < argc; ++i) if (f == argv[i]) return true;
    return false;
}
static bool parseKFromArgs(int argc, char** argv, Intrinsics& Kout) {
    for (int i = 0; i < argc; ++i) if (std::string(argv[i]) == "--K" && i+4 < argc) {
        Kout.fx = std::atof(argv[i+1]); Kout.fy = std::atof(argv[i+2]);
        Kout.cx = std::atof(argv[i+3]); Kout.cy = std::atof(argv[i+4]);
        Kout.K = (cv::Mat_<double>(3,3) << Kout.fx,0,Kout.cx, 0,Kout.fy,Kout.cy, 0,0,1);
        Kout.haveK = true; return true;
    }
    return false;
}
static void usage(){
    std::cout <<
    "Epipolar Geometry Visualizer\n"
    "Usage:\n"
    "  epipolar_viz <img1> <img2> [--K fx fy cx cy] [--teach | --gui]\n\n"
    "Modes:\n"
    "  --teach   Guided tutorial with Learn panel (default)\n"
    "  --gui     Minimal GUI only\n";
}

int main(int argc, char** argv){
    if (argc < 3) { usage(); return 1; }
    std::string p1=argv[1], p2=argv[2];
    if (!fs::exists(p1) || !fs::exists(p2)) { std::cerr<<"[error] bad image path\n"; return 1; }

    cv::Mat L=cv::imread(p1, cv::IMREAD_COLOR), R=cv::imread(p2, cv::IMREAD_COLOR);
    if (L.empty() || R.empty()) { std::cerr<<"[error] cannot read image(s)\n"; return 1; }

    Intrinsics K; parseKFromArgs(argc, argv, K);
    bool teach = hasFlag(argc, argv, "--teach");
    bool gui   = hasFlag(argc, argv, "--gui");

    if (!teach && !gui) teach = true; // default to tutorial
    if (teach) runTeach(L,R,K);
    else       runSandbox(L,R,K);
    return 0;
}
