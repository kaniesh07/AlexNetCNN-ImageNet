#!/usr/bin/env python3
"""
gen_vectors.py -- representative input vectors + expected outputs for
tb_rtl_stream_top.v, computed with pynq/rtl_stream_model.py (the same golden
model used on the board).  Writes, into the output directory:
    in_vec.hex    one line per input beat  : 9 hex digits {last, data[31:0]}
    exp_vec.hex   one line per output beat : 9 hex digits {last, data[31:0]}
    vec_params.vh `define N_IN / N_OUT for the testbench
If build/deploy/golden exists, one packet re-uses real AlexNet conv5
activations and is checked against the HLS kernel's (C-sim verified) output.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "pynq"))
import rtl_stream_model as m  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=HERE)
    ap.add_argument("--deploy", default=os.path.join(ROOT, "build", "deploy"))
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    np.random.seed(a.seed)
    rng = np.random.default_rng(a.seed)
    ri = lambda *s: rng.integers(-128, 128, size=s).astype(np.int8)   # noqa: E731

    packets = []
    # 1. pass-through and ReLU, including the extremes -128/127/0/-1
    edge = np.array([[-128, 127, 0, -1], [1, -2, -127, 126]], np.int8)
    packets.append(("relu-off", m.relu_packet(np.vstack([edge, ri(14, 4)]), 0)))
    packets.append(("relu-on", m.relu_packet(np.vstack([edge, ri(30, 4)]), 1)))
    # 2. max pooling windows (all-negative window, ties, random)
    wins = ri(20, 9, 4)
    wins[0] = -100
    wins[1] = 5
    packets.append(("pool", m.pool_packet(wins, 0)))
    packets.append(("pool-relu", m.pool_packet(ri(12, 9, 4), 1)))
    # 3. 3x3 convolution windows: raw accumulators and requantised
    packets.append(("conv-raw", m.conv_packet(ri(9), -123456, ri(16, 9), 0, 0, 0)))
    packets.append(("conv-raw-relu", m.conv_packet(ri(9), 1000, ri(16, 9), 0, 0, 1)))
    packets.append(("conv-rq-sh7", m.conv_packet(ri(9), 2000, ri(24, 9), 7, 1, 0)))
    packets.append(("conv-rq-sh9-relu", m.conv_packet(ri(9), -500, ri(24, 9), 9, 1, 1)))
    packets.append(("conv-saturate", m.conv_packet(np.full(9, 127, np.int8), 0,
                                                    np.full((4, 9), 127, np.int8), 1, 1, 0)))
    packets.append(("conv-sat-neg", m.conv_packet(np.full(9, -128, np.int8), 0,
                                                   np.full((4, 9), 127, np.int8), 2, 1, 0)))
    if os.path.isdir(os.path.join(a.deploy, "golden")):
        packets.append(("hls-golden-conv5-pool", m.golden_pool_packet(a.deploy, 4, 0)))
        print("included HLS golden cross-check packet (conv5 maxpool, channels 0-3)")

    ins, outs = [], []
    for name, (bi, bo) in packets:
        ins += bi
        outs += bo
        print(f"  {name:22s} in={len(bi):4d} beats  out={len(bo):4d} beats")
    with open(os.path.join(a.out, "in_vec.hex"), "w") as f:
        for d, l in ins:
            f.write(f"{int(l)}{d:08x}\n")
    with open(os.path.join(a.out, "exp_vec.hex"), "w") as f:
        for d, l in outs:
            f.write(f"{int(l)}{d:08x}\n")
    with open(os.path.join(a.out, "vec_params.vh"), "w") as f:
        f.write(f"`define N_IN {len(ins)}\n`define N_OUT {len(outs)}\n`define N_PKT {len(packets)}\n")
    print(f"total: {len(ins)} input beats, {len(outs)} output beats, {len(packets)} packets")


if __name__ == "__main__":
    main()
