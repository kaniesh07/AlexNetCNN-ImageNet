// alexnet_accel.cpp -- top level of the AlexNet layer engine (Vitis HLS 2024.2)
// See include/alexnet_accel.h for the full interface contract.
#include "alexnet_accel.h"
#include "engines.h"

static int validate(int op, int cin, int hin, int win, int cout, int ksize, int pad,
                    int hout, int wout, int pool, int raw_out) {
    if (op == OP_CONV) {
        if (cin <= 0 || cout <= 0 || (cin % TI) != 0 || (cout % TO) != 0) return ERR_CHANNELS;
        if (ksize < 1 || ksize > MAX_K || pad < 0 || pad >= ksize) return ERR_KERNEL;
        if (hin <= 0 || win <= 0) return ERR_KERNEL;
        if (hout != hin + 2 * pad - ksize + 1 || wout != win + 2 * pad - ksize + 1) return ERR_KERNEL;
        if (hout <= 0 || wout <= 0) return ERR_KERNEL;
        if (pool && (hout < 3 || wout < 3)) return ERR_KERNEL;
        if ((cin / TI) * hin * win > IN_BUF_WORDS) return ERR_CAPACITY;
        if ((cin / TI) * ksize * ksize > W_DEPTH) return ERR_CAPACITY;
        if (hout * wout > OUT_TILE_DEPTH) return ERR_CAPACITY;
        if (pool && ((hout - 3) / 2 + 1) * ((wout - 3) / 2 + 1) > POOL_DEPTH) return ERR_CAPACITY;
        if (cout > P_DEPTH) return ERR_CAPACITY;
        if (raw_out) return ERR_BAD_OP;            // raw output only for FC
        return ERR_OK;
    }
    if (op == OP_FC) {
        if (cin <= 0 || cout <= 0 || (cin % TI) != 0) return ERR_CHANNELS;
        if (raw_out ? (cout % 2) != 0 : (cout % 8) != 0) return ERR_CHANNELS;
        if (cin / TI > IN_BUF_WORDS || cout > P_DEPTH) return ERR_CAPACITY;
        if (pool) return ERR_BAD_OP;
        return ERR_OK;
    }
    return ERR_BAD_OP;
}

int alexnet_accel(const word_t *act_in, word_t *act_out,
                  const word_t *weights, const word_t *params,
                  int op, int cin, int hin, int win, int cout,
                  int ksize, int pad, int hout, int wout,
                  int pool, int relu, int raw_out) {
// depth=8192 only sizes the C/RTL co-simulation memories (small test layers);
// it has no effect on the hardware.
#pragma HLS INTERFACE mode=m_axi port=act_in  bundle=gmem_act offset=slave depth=8192 max_read_burst_length=256 max_write_burst_length=256 num_read_outstanding=8 num_write_outstanding=8
#pragma HLS INTERFACE mode=m_axi port=act_out bundle=gmem_act offset=slave depth=8192
#pragma HLS INTERFACE mode=m_axi port=weights bundle=gmem_w   offset=slave depth=8192 max_read_burst_length=256 num_read_outstanding=16
#pragma HLS INTERFACE mode=m_axi port=params  bundle=gmem_w   offset=slave depth=8192
#pragma HLS INTERFACE mode=s_axilite port=act_in   bundle=control
#pragma HLS INTERFACE mode=s_axilite port=act_out  bundle=control
#pragma HLS INTERFACE mode=s_axilite port=weights  bundle=control
#pragma HLS INTERFACE mode=s_axilite port=params   bundle=control
#pragma HLS INTERFACE mode=s_axilite port=op       bundle=control
#pragma HLS INTERFACE mode=s_axilite port=cin      bundle=control
#pragma HLS INTERFACE mode=s_axilite port=hin      bundle=control
#pragma HLS INTERFACE mode=s_axilite port=win      bundle=control
#pragma HLS INTERFACE mode=s_axilite port=cout     bundle=control
#pragma HLS INTERFACE mode=s_axilite port=ksize    bundle=control
#pragma HLS INTERFACE mode=s_axilite port=pad      bundle=control
#pragma HLS INTERFACE mode=s_axilite port=hout     bundle=control
#pragma HLS INTERFACE mode=s_axilite port=wout     bundle=control
#pragma HLS INTERFACE mode=s_axilite port=pool     bundle=control
#pragma HLS INTERFACE mode=s_axilite port=relu     bundle=control
#pragma HLS INTERFACE mode=s_axilite port=raw_out  bundle=control
#pragma HLS INTERFACE mode=s_axilite port=return   bundle=control

    // ---- on-chip buffers (BRAM) -----------------------------------------
    static word_t  in_buf[IN_BUF_WORDS];       // whole input feature map / FC vector
    static word_t  p_buf[P_DEPTH];             // per-channel bias|M|shift
    static word_t  w_buf[TO][W_DEPTH];         // weights of the current 16-ch tile
    static word2_t out_tile[OUT_TILE_DEPTH];   // requantised conv outputs of the tile
    static word2_t pooled[POOL_DEPTH];         // pooled outputs of the tile
    word_t tile_params[TO];
#pragma HLS ARRAY_PARTITION variable=w_buf complete dim=1
#pragma HLS ARRAY_PARTITION variable=tile_params complete dim=1
#pragma HLS BIND_STORAGE variable=in_buf   type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=out_tile type=ram_2p impl=bram

    const int err = validate(op, cin, hin, win, cout, ksize, pad, hout, wout, pool, raw_out);
    if (err != ERR_OK) return err;

    if (op == OP_FC) {
        load_words(act_in, in_buf, cin / TI);
        load_params(params, p_buf, cout);
        fc_layer(in_buf, p_buf, weights, act_out, cin / TI, cout, relu, raw_out);
        return ERR_OK;
    }

    // ---- convolution: time-multiplexed over 16-channel output tiles -----
    const int cig   = cin / TI;
    const int cigkk = cig * ksize * ksize;
    const int hp    = pool ? (hout - 3) / 2 + 1 : hout;
    const int wp    = pool ? (wout - 3) / 2 + 1 : wout;
    const int npix  = hp * wp;

    load_words(act_in, in_buf, cig * hin * win);
    load_params(params, p_buf, cout);

TILES:
    for (int t = 0; t < cout / TO; t++) {
#pragma HLS LOOP_TRIPCOUNT min=4 max=24 avg=16
        load_conv_weights(weights + t * TO * cigkk, w_buf, cigkk);
    TP:
        for (int i = 0; i < TO; i++) {
#pragma HLS PIPELINE II=1
            tile_params[i] = p_buf[t * TO + i];
        }
        conv_tile_mac(in_buf, w_buf, tile_params, out_tile,
                      cig, hin, win, ksize, pad, hout, wout, relu);
        if (pool) maxpool_tile(out_tile, pooled, wout, hp, wp);
        store_conv_tile(act_out + (2 * t) * npix, out_tile, pooled, pool, npix);
    }
    return ERR_OK;
}
