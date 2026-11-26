#!/usr/bin/env python3
"""
arm_reference.py -- ARM-only baseline: the SAME int8 network (same blobs, same
requantisation) executed with NumPy on the Cortex-A9.  Bit-exact with the
FPGA, so it doubles as an on-board verification oracle and as the "ARM-only
inference" baseline required for Stage E.

Memory: FC weights are processed in row chunks so peak RAM stays < ~40 MB.
Usage:  python3 arm_reference.py --deploy deploy --image cat.jpg
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import alexnet_common as ac  # noqa: E402


class ArmInt8Model:
    def __init__(self, deploy):
        self.layers = []
        for i, L in enumerate(ac.HW_LAYERS):
            w = np.load(os.path.join(deploy, f"layer{i}_weights.npy"), mmap_mode="r")
            b, M, sh = ac.unpack_params(np.load(os.path.join(deploy, f"layer{i}_params.npy")))
            if L["op"] == ac.OP_CONV:
                wq = ac.unpack_conv_weights(np.array(w), L["cout"], L["cin"], L["k"])
            else:
                wq = np.asarray(w).view(np.int8).reshape(L["cout"], L["cin"])   # kernel input order
            self.layers.append((L, wq, b, M, sh))

    def forward(self, input_words, chunk=256, timing=None):
        x = input_words
        for L, wq, b, M, sh in self.layers:
            t0 = time.perf_counter()
            if L["op"] == ac.OP_CONV:
                x = ac.run_layer_int(L, x, wq, b, M, sh)
            else:
                v = np.ascontiguousarray(x[: L["cin"] // 8]).view(np.int8).astype(np.float64)
                acc = np.empty(L["cout"], np.int64)
                for r in range(0, L["cout"], chunk):
                    acc[r:r + chunk] = np.rint(wq[r:r + chunk].astype(np.float64) @ v).astype(np.int64)
                acc += b
                if L["raw"]:
                    x = np.ascontiguousarray(acc.astype(np.int32)).view(np.uint64).copy()
                else:
                    x = np.ascontiguousarray(ac.requant(acc, M, sh, L["relu"])).view(np.uint64).copy()
            if timing is not None:
                timing[L["name"]] = time.perf_counter() - t0
        return ac.unpack_raw_i32(x, 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deploy", default=os.path.join(HERE, "deploy"))
    ap.add_argument("--image")
    ap.add_argument("--labels", default=os.path.join(HERE, "imagenet_classes.txt"))
    a = ap.parse_args()
    import json
    meta = json.load(open(os.path.join(a.deploy, "model_meta.json")))
    if a.image:
        from preprocessing import load_and_prepare
        _, words, _ = load_and_prepare(a.image, meta["input_scale"])
    else:
        words = np.load(os.path.join(a.deploy, "golden_input.npy"))
    m = ArmInt8Model(a.deploy)
    t = {}
    t0 = time.perf_counter()
    raw = m.forward(words, timing=t)
    total = time.perf_counter() - t0
    for k, v in t.items():
        print(f"  {k:6s} {v*1e3:10.1f} ms")
    print(f"ARM-only int8 total: {total*1e3:.1f} ms")
    if not a.image:
        exp = ac.unpack_raw_i32(np.load(os.path.join(a.deploy, "golden_fc8_raw.npy")), 1000)
        print("golden fc8 bit-exact:", np.array_equal(exp, raw))
    lg = raw * np.load(os.path.join(a.deploy, "fc8_logit_scale.npy"))
    labels = open(a.labels).read().splitlines()
    print("top-1:", int(np.argmax(lg)), labels[int(np.argmax(lg))])


if __name__ == "__main__":
    main()
