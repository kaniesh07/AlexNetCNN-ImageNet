// ============================================================================
// alexnet_accel.h -- Vitis HLS 2024.2 AlexNet layer engine for the XC7Z020
//
// ONE runtime-configurable engine executes every AlexNet layer, one layer per
// invocation (time-multiplexed).  The PS (PYNQ) sequences the 8 layers.
//
// INTERFACE CONTRACT ---------------------------------------------------------
//  AXI4-Lite slave  "control"  (s_axi_control): ap_ctrl_hs block-level control
//      + all scalar arguments + 64-bit buffer base addresses + ap_return.
//      Register OFFSETS are assigned by Vitis HLS; never hard-code them --
//      read them from the generated header
//        <solution>/impl/misc/drivers/alexnet_accel_v1_0/src/xalexnet_accel_hw.h
//      (pynq/hw_header.py parses this file at run time).
//  AXI4 master "gmem_act" (m_axi_gmem_act, 64-bit data):  act_in (read),
//      act_out (write)  -> PS S_AXI_HP0
//  AXI4 master "gmem_w"   (m_axi_gmem_w,   64-bit data):  weights, params
//      (read only)      -> PS S_AXI_HP1
//
// DATA FORMATS ---------------------------------------------------------------
//  word_t = 64-bit little-endian word.
//  Activations "C8HW": word[(c/8)*H*W + y*W + x], byte lane (c%8) holds
//      int8 activation of channel c.  C must be a multiple of 8.
//      FC vectors are the same layout with H=W=1 (lane l of word j = elem 8j+l).
//  Conv weights: words in order [Cout/16 tiles][16][Cin/8][K][K];
//      lane l = int8 weight of input channel (cig*8+l).
//  FC weights:   words in order [Cout][Cin/8]; lane l = int8 weight for
//      input element 8j+l of the packed input vector.
//  Params: one word per output channel:
//      [31:0]  bias   int32 (in accumulator units)
//      [47:32] M      uint16 requant multiplier (0..32767)
//      [53:48] sh     6-bit requant shift (1..47)
//  Requantisation (all non-raw layers):
//      y = sat_int8( (acc*M + 2^(sh-1)) >> sh ),  ReLU if relu!=0
//  Raw output (raw_out=1, FC only): int32 accumulators, 2 per word
//      (neuron 2i in [31:0], 2i+1 in [63:32]).
//  Pooling (pool=1, conv only): 3x3, stride 2, no padding, applied after
//      requant/ReLU (exact: requant is monotonic increasing since M>=0).
//
// RETURN (ap_return): 0 = OK, otherwise ERR_* below; on error nothing is
//      written to act_out.
// ============================================================================
#ifndef ALEXNET_ACCEL_H
#define ALEXNET_ACCEL_H

#include <ap_int.h>

typedef ap_uint<64>  word_t;
typedef ap_uint<128> word2_t;   // 16 int8 lanes (one conv output tile pixel)
typedef ap_int<32>   acc_t;

// --- compile-time capacity (must match pynq/alexnet_common.py) -------------
static const int TI             = 8;      // channels per word
static const int TO             = 16;     // parallel output channels (conv)
static const int MAX_K          = 5;
static const int IN_BUF_WORDS   = 19494;  // 6*57*57 words (conv1' input)
static const int W_DEPTH        = 432;    // max (Cin/8)*K*K  (conv4)
static const int OUT_TILE_DEPTH = 3025;   // max Hout*Wout    (55*55)
static const int POOL_DEPTH     = 729;    // max pooled pixels (27*27)
static const int P_DEPTH        = 4096;   // max Cout

// --- op codes / status -------------------------------------------------------
static const int OP_CONV = 0;
static const int OP_FC   = 1;

static const int ERR_OK        = 0;
static const int ERR_BAD_OP    = 1;
static const int ERR_CHANNELS  = 2;   // cin%8 / cout%16 (conv) / cout%8 (fc)
static const int ERR_KERNEL    = 3;   // ksize/pad/out size inconsistent
static const int ERR_CAPACITY  = 4;   // exceeds on-chip buffer sizes

int alexnet_accel(const word_t *act_in, word_t *act_out,
                  const word_t *weights, const word_t *params,
                  int op, int cin, int hin, int win, int cout,
                  int ksize, int pad, int hout, int wout,
                  int pool, int relu, int raw_out);

#endif
