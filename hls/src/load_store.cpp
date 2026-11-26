// load_store.cpp -- DDR <-> BRAM movers (burst-friendly sequential loops)
#include "engines.h"

// Sequential burst read of n words into the input-activation buffer.
void load_words(const word_t *src, word_t dst[IN_BUF_WORDS], int n) {
LOAD_ACT:
    for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1152 max=19494 avg=8000
        dst[i] = src[i];
    }
}

// Per-output-channel parameter words (bias | M | shift) for a whole layer.
void load_params(const word_t *src, word_t dst[P_DEPTH], int n) {
LOAD_PARAMS:
    for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=64 max=4096 avg=1000
        dst[i] = src[i];
    }
}

// Weights of one 16-output-channel tile: TO*cigkk contiguous words.
// Written round-robin into 16 banks (w_buf is completely partitioned on dim 1).
void load_conv_weights(const word_t *src, word_t w_buf[TO][W_DEPTH], int cigkk) {
#pragma HLS ARRAY_PARTITION variable=w_buf complete dim=1
    int to = 0, i = 0;
    const int n = TO * cigkk;
LOAD_W:
    for (int k = 0; k < n; k++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=864 max=6912 avg=4000
        word_t v = src[k];
        w_buf[to][i] = v;
        if (i == cigkk - 1) { i = 0; to++; } else { i++; }
    }
}

// Write one finished conv tile (16 channels = 2 channel-groups) to DDR in
// C8HW layout.  dst_base points at channel-group (2*tile).
void store_conv_tile(word_t *dst_base, const word2_t src[OUT_TILE_DEPTH],
                     const word2_t pooled[POOL_DEPTH], int use_pool, int npix) {
STORE_GROUP:
    for (int g = 0; g < 2; g++) {
        word_t *dst = dst_base + g * npix;
    STORE_PIX:
        for (int p = 0; p < npix; p++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=36 max=3025 avg=500
            word2_t v = use_pool ? pooled[p] : src[p];
            dst[p] = (g == 0) ? word_t(v.range(63, 0)) : word_t(v.range(127, 64));
        }
    }
}
