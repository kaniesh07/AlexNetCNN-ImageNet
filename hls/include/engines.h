// engines.h -- internal sub-function prototypes of the AlexNet layer engine
#ifndef ENGINES_H
#define ENGINES_H

#include "alexnet_accel.h"

// load_store.cpp ---------------------------------------------------------------
void load_words(const word_t *src, word_t dst[IN_BUF_WORDS], int n);
void load_params(const word_t *src, word_t dst[P_DEPTH], int n);
void load_conv_weights(const word_t *src, word_t w_buf[TO][W_DEPTH], int cigkk);
void store_conv_tile(word_t *dst_base, const word2_t src[OUT_TILE_DEPTH],
                     const word2_t pooled[POOL_DEPTH], int use_pool, int npix);

// conv_engine.cpp --------------------------------------------------------------
void conv_tile_mac(const word_t in_buf[IN_BUF_WORDS], const word_t w_buf[TO][W_DEPTH],
                   const word_t tile_params[TO], word2_t out_tile[OUT_TILE_DEPTH],
                   int cig, int hin, int win, int ksize, int pad,
                   int hout, int wout, int relu);
void maxpool_tile(const word2_t out_tile[OUT_TILE_DEPTH], word2_t pooled[POOL_DEPTH],
                  int wout, int hp, int wp);

// fc_engine.cpp ----------------------------------------------------------------
void fc_layer(const word_t in_buf[IN_BUF_WORDS], const word_t p_buf[P_DEPTH],
              const word_t *weights, word_t *act_out,
              int cinw, int cout, int relu, int raw_out);

#endif
