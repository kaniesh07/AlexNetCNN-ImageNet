// quant_utils.h -- quantisation / arithmetic helpers (all INLINE)
#ifndef QUANT_UTILS_H
#define QUANT_UTILS_H

#include "alexnet_accel.h"

// 8-lane signed int8 dot product.  |sum| <= 8*128*128 = 2^17 -> 19 bits.
static inline ap_int<20> dot8(word_t w, word_t x) {
#pragma HLS INLINE
    ap_int<20> s = 0;
DOT8:
    for (int l = 0; l < TI; l++) {
#pragma HLS UNROLL
        ap_int<8> a = w.range(8 * l + 7, 8 * l);
        ap_int<8> b = x.range(8 * l + 7, 8 * l);
        ap_int<16> p = a * b;
        s += p;
    }
    return s;
}

// Param word fields
static inline acc_t       param_bias(word_t p)  { return (acc_t)(ap_int<32>)p.range(31, 0); }
static inline ap_uint<16> param_mult(word_t p)  { return p.range(47, 32); }
static inline ap_uint<6>  param_shift(word_t p) { return p.range(53, 48); }

// ReLU on one int8
static inline ap_int<8> relu8(ap_int<8> v) {
#pragma HLS INLINE
    return (v < 0) ? ap_int<8>(0) : v;
}

// y = sat8((acc*M + 2^(sh-1)) >> sh), optional ReLU.
// |acc*M| < 2^31*2^15 = 2^46 and rounding term < 2^46 -> fits ap_int<48>.
static inline ap_int<8> requant(acc_t acc, ap_uint<16> m, ap_uint<6> sh, bool relu) {
#pragma HLS INLINE
    ap_int<48> p   = (ap_int<48>)acc * (ap_int<17>)m;
    ap_uint<6> shm = (sh == 0) ? ap_uint<6>(1) : sh;      // sh==0 is illegal -> treat as 1
    ap_int<48> rnd = ((ap_int<48>)1) << (shm - 1);
    ap_int<48> r   = (p + rnd) >> shm;                    // arithmetic shift
    ap_int<8> y;
    if (r > 127)       y = 127;
    else if (r < -128) y = -128;
    else               y = (ap_int<8>)r;
    return relu ? relu8(y) : y;
}

// Lane-wise signed max of 16 int8 lanes
static inline word2_t lanemax16(word2_t a, word2_t b) {
#pragma HLS INLINE
    word2_t r;
    for (int l = 0; l < 2 * TI; l++) {
#pragma HLS UNROLL
        ap_int<8> x = a.range(8 * l + 7, 8 * l);
        ap_int<8> y = b.range(8 * l + 7, 8 * l);
        r.range(8 * l + 7, 8 * l) = (x > y) ? x : y;
    }
    return r;
}

#endif
