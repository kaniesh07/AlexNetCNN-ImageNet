// fc_engine.cpp -- fully connected layer, weights streamed from DDR
//
// The FC layers are memory-bandwidth bound (fc6 alone = 37.7 MB of int8
// weights, each used once per image).  The engine consumes one 64-bit weight
// word (8 MACs) per clock in a single long sequential burst on gmem_w; the
// input vector sits in BRAM.  At 100 MHz that is 800 MB/s peak demand, i.e.
// at or above what one Zynq-7000 HP port sustains in practice.
#include "engines.h"
#include "quant_utils.h"

void fc_layer(const word_t in_buf[IN_BUF_WORDS], const word_t p_buf[P_DEPTH],
              const word_t *weights, word_t *act_out,
              int cinw, int cout, int relu, int raw_out) {
    const int total = cout * cinw;
    const bool do_relu = (relu != 0);
    int j = 0, o = 0;
    acc_t acc = 0;
    word_t pack = 0;
FC:
    for (int n = 0; n < total; n++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=512000 max=4718592 avg=2000000
        const word_t w = weights[n];
        const word_t pw = p_buf[o];
        const acc_t base = (j == 0) ? param_bias(pw) : acc;
        const acc_t a = base + (acc_t)dot8(w, in_buf[j]);
        acc = a;
        if (j == cinw - 1) {
            if (raw_out) {
                if (o & 1) {
                    pack.range(63, 32) = a;
                    act_out[o >> 1] = pack;
                } else {
                    pack.range(31, 0) = a;
                }
            } else {
                const int lane = o & 7;
                pack.range(8 * lane + 7, 8 * lane) = requant(a, param_mult(pw), param_shift(pw), do_relu);
                if (lane == 7) act_out[o >> 3] = pack;
            }
            j = 0;
            o++;
        } else {
            j++;
        }
    }
}
