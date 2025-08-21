#pragma once
struct MatchParams {
    int   num_features = 2500;
    float ratio        = 0.75f;
    bool  do_symmetric = false;
    int   max_draw     = 150;
};
struct RansacParams {
    double F_px_thresh   = 1.0;
    double F_conf        = 0.999;
    double E_norm_thresh = 0.0015;
    double E_conf        = 0.999;
};

