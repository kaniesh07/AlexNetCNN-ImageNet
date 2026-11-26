// conv_engine.cpp -- stride-1 KxK convolution tile engine + 3x3/s2 max pool
//
// One call computes 16 output channels over the whole output plane.
// Inner datapath: 16 (TO) x 8 (TI) = 128 int8 MACs per clock, II = 1.
// The (oy, ox, cig, ky, kx) nest is flattened into a single pipelined loop;
// all address arithmetic is incremental (no runtime multipliers).
#include "engines.h"
#include "quant_utils.h"

void conv_tile_mac(const word_t in_buf[IN_BUF_WORDS], const word_t w_buf[TO][W_DEPTH],
                   const word_t tile_params[TO], word2_t out_tile[OUT_TILE_DEPTH],
                   int cig, int hin, int win, int ksize, int pad,
                   int hout, int wout, int relu) {
#pragma HLS ARRAY_PARTITION variable=w_buf complete dim=1
#pragma HLS ARRAY_PARTITION variable=tile_params complete dim=1

    acc_t       acc[TO];
    acc_t       bias[TO];
    ap_uint<16> mult[TO];
    ap_uint<6>  shft[TO];
#pragma HLS ARRAY_PARTITION variable=acc  complete
#pragma HLS ARRAY_PARTITION variable=bias complete
#pragma HLS ARRAY_PARTITION variable=mult complete
#pragma HLS ARRAY_PARTITION variable=shft complete

INIT:
    for (int t = 0; t < TO; t++) {
#pragma HLS UNROLL
        bias[t] = param_bias(tile_params[t]);
        mult[t] = param_mult(tile_params[t]);
        shft[t] = param_shift(tile_params[t]);
        acc[t]  = 0;
    }

    const int kk     = ksize * ksize;
    const int cigkk  = cig * kk;
    const int hw_in  = hin * win;
    const int total  = hout * wout * cigkk;
    const bool do_relu = (relu != 0);

    // incremental counters
    int ox = 0, oy = 0, c = 0, ky = 0, kx = 0, widx = 0, pix = 0;
    int cbase = 0;              // c*hw_in
    int kyoff = 0;              // ky*win
    int oyb   = -pad * win;     // (oy-pad)*win

MAC:
    for (int n = 0; n < total; n++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=650000 max=1750000 avg=1000000
        const int iy = oy - pad + ky;
        const int ix = ox - pad + kx;
        const bool valid = (iy >= 0) && (iy < hin) && (ix >= 0) && (ix < win);
        word_t x = 0;                                   // zero padding
        if (valid) x = in_buf[cbase + oyb + kyoff + ix];

        const bool first = (widx == 0);
        const bool last  = (widx == cigkk - 1);
        word2_t res = 0;
    LANES:
        for (int t = 0; t < TO; t++) {
#pragma HLS UNROLL
            acc_t base = first ? bias[t] : acc[t];
            acc_t a    = base + (acc_t)dot8(w_buf[t][widx], x);
            acc[t]     = a;
            res.range(8 * t + 7, 8 * t) = requant(a, mult[t], shft[t], do_relu);
        }
        if (last) out_tile[pix] = res;

        // ---- advance the flattened nest --------------------------------
        if (last) {
            widx = 0; kx = 0; ky = 0; kyoff = 0; c = 0; cbase = 0;
            pix++;
            if (ox == wout - 1) { ox = 0; oy++; oyb += win; } else { ox++; }
        } else {
            widx++;
            if (kx == ksize - 1) {
                kx = 0;
                if (ky == ksize - 1) { ky = 0; kyoff = 0; c++; cbase += hw_in; }
                else                 { ky++; kyoff += win; }
            } else {
                kx++;
            }
        }
    }
}

// 3x3 window, stride 2, no padding, on a 16-lane tile (lane-wise max).
void maxpool_tile(const word2_t out_tile[OUT_TILE_DEPTH], word2_t pooled[POOL_DEPTH],
                  int wout, int hp, int wp) {
    const int total = hp * wp * 9;
    int px = 0, wy = 0, wx = 0, pidx = 0;
    int rowb = 0;      // 2*py*wout
    int wyoff = 0;     // wy*wout
    word2_t m = 0;
POOL:
    for (int n = 0; n < total; n++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=324 max=6561 avg=2000
        word2_t v = out_tile[rowb + wyoff + 2 * px + wx];
        const bool first = (wy == 0) && (wx == 0);
        const bool last  = (wy == 2) && (wx == 2);
        m = first ? v : lanemax16(m, v);
        if (last) {
            pooled[pidx] = m;
            pidx++;
            wy = 0; wx = 0; wyoff = 0;
            if (px == wp - 1) { px = 0; rowb += 2 * wout; } else { px++; }
        } else if (wx == 2) {
            wx = 0; wy++; wyoff += wout;
        } else {
            wx++;
        }
    }
}
