#!/usr/bin/env python3
"""
alexnet_inference.py -- end-to-end AlexNet classification on the PYNQ-Z2.

Usage (on the board, as root / with sudo, from the directory holding the files):
  sudo python3 alexnet_inference.py --image cat.jpg
  sudo python3 alexnet_inference.py --verify-golden              # Stage D check
  sudo python3 alexnet_inference.py --image cat.jpg --compare-arm  # FPGA vs ARM-only
Directory layout expected (see docs/deployment.md):
  overlay/alexnet.bit  overlay/alexnet.hwh  overlay/xalexnet_accel_hw.h
  deploy/layer*_weights.npy layer*_params.npy fc8_logit_scale.npy model_meta.json
  deploy/golden_input.npy deploy/golden/golden*.bin
  imagenet_classes.txt
No internet access is needed at run time.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import alexnet_common as ac          # noqa: E402
from accelerator_driver import AlexNetAccel   # noqa: E402


def load_labels(path):
    with open(path) as f:
        labels = [l.rstrip("\n") for l in f]
    if len(labels) != 1000:
        raise SystemExit(f"{path}: expected 1000 labels, got {len(labels)}")
    return labels


def top5(logits, labels):
    p = ac.softmax(logits.astype(np.float64))
    idx = np.argsort(-p)[:5]
    return [(int(i), labels[i], float(p[i])) for i in idx]


def verify_golden(acc, deploy):
    """Stage D: run the golden input and compare every layer bit-exactly."""
    inp = np.load(os.path.join(deploy, "golden_input.npy"))
    _, timing, inter = acc.infer(inp, keep_intermediates=True)
    ok = True
    for i, (L, got) in enumerate(zip(ac.HW_LAYERS, inter)):
        exp = np.fromfile(os.path.join(deploy, "golden", f"golden{i}.bin"), dtype=np.uint64)
        bad = int(np.count_nonzero(got != exp[: got.size]))
        ok &= bad == 0
        print(f"  {L['name']:6s} words={got.size:6d} mismatches={bad:6d}  "
              f"t={timing[L['name']]*1e3:8.2f} ms  {'PASS' if bad == 0 else 'FAIL'}")
    print("GOLDEN:", "PASS (bit-exact)" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlay", default=os.path.join(HERE, "overlay", "alexnet.bit"))
    ap.add_argument("--header", default=os.path.join(HERE, "overlay", "xalexnet_accel_hw.h"))
    ap.add_argument("--deploy", default=os.path.join(HERE, "deploy"))
    ap.add_argument("--labels", default=os.path.join(HERE, "imagenet_classes.txt"))
    ap.add_argument("--image", nargs="*", default=[])
    ap.add_argument("--repeat", type=int, default=1, help="timed repetitions per image")
    ap.add_argument("--timeout", type=float, default=5.0, help="per-layer timeout (s)")
    ap.add_argument("--verify-golden", action="store_true")
    ap.add_argument("--compare-arm", action="store_true",
                    help="also run the bit-exact int8 model on the ARM (slow) and compare")
    args = ap.parse_args()

    from pynq import Overlay
    from preprocessing import load_and_prepare

    meta = json.load(open(os.path.join(args.deploy, "model_meta.json")))
    logit_scale = np.load(os.path.join(args.deploy, "fc8_logit_scale.npy"))
    labels = load_labels(args.labels)

    t0 = time.perf_counter()
    ol = Overlay(args.overlay)            # loads alexnet.bit, parses alexnet.hwh
    t_ol = time.perf_counter() - t0
    print(f"overlay loaded in {t_ol:.2f} s; IPs: {sorted(ol.ip_dict)}")
    header = args.header if os.path.exists(args.header) else None

    with AlexNetAccel(ol, args.deploy, header, args.timeout) as acc:
        print(f"accelerator '{acc.name}' register map from {acc.regs.source}")
        acc.self_test_registers()
        print(f"model blobs copied to DDR in {acc.load_time_s:.2f} s")
        if args.verify_golden:
            verify_golden(acc, args.deploy)
        for path in args.image:
            for r in range(args.repeat):
                x, words, tpre = load_and_prepare(path, meta["input_scale"])
                raw, timing, _ = acc.infer(words)
                t_post0 = time.perf_counter()
                logits = raw.astype(np.float64) * logit_scale
                res = top5(logits, labels)
                t_post = time.perf_counter() - t_post0
            print(f"\n{path}")
            for i, lab, p in res:
                print(f"  {i:4d}  {p*100:6.2f}%  {lab}")
            total = tpre["preprocess"] + tpre["quantize_pack"] + timing["end_to_end"] + t_post
            print("  timing (last repetition):")
            print(f"    preprocess (CPU)           {tpre['preprocess']*1e3:9.2f} ms")
            print(f"    quantize+s2d+pack (CPU)    {tpre['quantize_pack']*1e3:9.2f} ms")
            print(f"    input copy to DMA buffer   {timing['input_copy']*1e3:9.2f} ms")
            for L in ac.HW_LAYERS:
                print(f"    {L['name']:6s} (FPGA)              {timing[L['name']]*1e3:9.2f} ms")
            print(f"    accelerator total          {timing['accelerator_total']*1e3:9.2f} ms")
            print(f"    output copy                {timing['output_copy']*1e3:9.2f} ms")
            print(f"    dequant+softmax+top5 (CPU) {t_post*1e3:9.2f} ms")
            print(f"    END-TO-END                 {total*1e3:9.2f} ms")
            if args.compare_arm:
                from arm_reference import ArmInt8Model
                arm = ArmInt8Model(args.deploy)
                ta = time.perf_counter()
                raw_arm = arm.forward(words)
                ta = time.perf_counter() - ta
                same = np.array_equal(raw_arm, raw)
                print(f"  ARM-only int8 (NumPy) : {ta*1e3:.1f} ms; fc8 bit-exact vs FPGA: {same}")
    if hasattr(ol, "free"):
        ol.free()


if __name__ == "__main__":
    main()
