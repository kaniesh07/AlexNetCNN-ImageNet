#!/usr/bin/env python3
"""
quantize_model.py -- HOST PC script (numpy + Pillow only).

Post-training int8 quantisation of torchvision AlexNet for the PYNQ-Z2 kernel
and generation of every file the board and the HLS testbench need.

Scheme (identical to hls/src/alexnet_accel.cpp and pynq/alexnet_common.py)
  activations : int8, symmetric, one scale per tensor  (s_a = max|a|/127)
  weights     : int8, symmetric, per OUTPUT channel    (s_w[o] = max|W_o|/127)
  bias        : int32, b_q = round(b / (s_in * s_w[o]))
  accumulator : int32  (worst case |acc| < 2^31 verified below)
  requant     : out = sat8((acc*M[o] + 2^(sh[o]-1)) >> sh[o]),  M in [2^14,2^15)
                M/2^sh ~= s_in*s_w[o]/s_out      (round half up, saturate)
  fc8         : raw int32 logits, logit = acc * s_in * s_w[o]   (CPU dequant)

Outputs (default build/deploy/)
  layer{i}_weights.npy, layer{i}_params.npy   uint64 blobs, DMA-ready
  fc8_logit_scale.npy                          float32[1000]
  model_meta.json                              scales + layer table
  golden/                                      raw .bin files for the HLS C testbench
                                               (input, per-layer weights/params/outputs)
Usage
  python model/quantize_model.py --calib-dir calib_images/ --golden-image cat.jpg
  python model/quantize_model.py --calib-random 8            # synthetic calibration
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "pynq"))
import alexnet_common as ac  # noqa: E402

ACT_NAMES = ["input", "conv1", "conv2", "conv3", "conv4", "conv5", "fc6", "fc7"]


def load_calibration(args):
    xs = []
    if args.calib_dir:
        from PIL import Image
        files = sorted(glob.glob(os.path.join(args.calib_dir, "*")))
        files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".jpeg"))]
        files = files[: args.calib_max]
        if not files:
            raise SystemExit("no images in --calib-dir")
        for f in files:
            xs.append(ac.preprocess_pil(Image.open(f)))
        print(f"calibration: {len(xs)} images from {args.calib_dir}")
    else:
        rng = np.random.default_rng(args.seed)
        for _ in range(args.calib_random):
            # smooth random "images" in pixel space, then normalised
            img = rng.random((3, 28, 28)).repeat(8, 1).repeat(8, 2)
            xs.append(((img.transpose(1, 2, 0) - ac.MEAN) / ac.STD).transpose(2, 0, 1).astype(np.float32))
        print(f"calibration: {len(xs)} SYNTHETIC images (pipeline test only)")
    return xs


def calibrate(P, xs, percentile):
    stats = {n: [] for n in ACT_NAMES}

    def rec(name, t):
        stats[name].append(np.abs(t).reshape(-1))
    for x in xs:
        stats["input"].append(np.abs(x).reshape(-1))
        ac.alexnet_float(x, P, rec)
    scales = {}
    for n in ACT_NAMES:
        v = np.concatenate(stats[n])
        mx = np.percentile(v, percentile) if percentile < 100 else v.max()
        scales[n] = float(max(mx, 1e-8) / 127.0)
    return scales


def quantize(P, scales):
    """Returns list of per-layer dicts with hardware-form int tensors."""
    layers = []
    in_names = ["input", "conv1", "conv2", "conv3", "conv4", "conv5", "fc6", "fc7"]
    out_names = ["conv1", "conv2", "conv3", "conv4", "conv5", "fc6", "fc7", None]
    for i, L in enumerate(ac.HW_LAYERS):
        key = ac.TORCH_KEYS[L["name"]]
        W = P[key + ".weight"].astype(np.float64)
        b = P[key + ".bias"].astype(np.float64)
        O = W.shape[0]
        wmax = np.abs(W.reshape(O, -1)).max(axis=1)
        s_w = np.where(wmax > 0, wmax / 127.0, 1e-12)
        wq = np.clip(np.rint(W / s_w.reshape((-1,) + (1,) * (W.ndim - 1))), -127, 127).astype(np.int8)
        s_in = scales[in_names[i]]
        bq = np.rint(b / (s_in * s_w))
        assert np.all(np.abs(bq) < 2 ** 31 - 1), f"{L['name']}: bias overflows int32"
        bq = bq.astype(np.int64)
        if out_names[i] is None:
            M = np.zeros(O, np.int64)
            sh = np.ones(O, np.int64)
            s_out = None
        else:
            s_out = scales[out_names[i]]
            M, sh = ac.choose_multiplier(s_in * s_w / s_out)
        # worst-case accumulator bound (inputs |x|<=128)
        fan_in = np.abs(wq.reshape(O, -1).astype(np.int64)).sum(axis=1)
        worst = (fan_in * 128 + np.abs(bq)).max()
        assert worst < 2 ** 31, f"{L['name']}: int32 accumulator may overflow"

        perm = None
        if L["op"] == ac.OP_CONV:
            w_hw = ac.space_to_depth_conv1_weights(wq) if L["name"] == "conv1" else wq
            words = ac.pack_conv_weights(w_hw)
        else:
            w_hw = wq
            if L["name"] == "fc6":
                perm = ac.fc_input_permutation(9216, 256, 6, 6)
            else:
                perm = ac.fc_input_permutation(L["cin"])
            words = ac.pack_fc_weights(wq, perm)
        layers.append(dict(L=L, wq=w_hw, bias=bq, M=M, sh=sh, s_w=s_w, s_in=s_in,
                           s_out=s_out, words=words, perm=perm,
                           params=ac.pack_params(bq, M, sh), worst_acc=int(worst)))
    return layers


def run_int(layers, x_float, s_in0):
    """Bit-exact integer forward pass on packed words. Returns per-layer outputs."""
    outs = []
    w = ac.prepare_input(x_float, s_in0)
    inp = w
    for q in layers:
        w = ac.run_layer_int(q["L"], w, q["wq"], q["bias"], q["M"], q["sh"], q["perm"])
        outs.append(w)
    return inp, outs


def logits_from_raw(words, layers):
    acc = ac.unpack_raw_i32(words, 1000).astype(np.float64)
    q = layers[-1]
    return (acc * q["s_in"] * q["s_w"]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(ROOT, "build", "alexnet_fp32.npz"))
    ap.add_argument("--out", default=os.path.join(ROOT, "build", "deploy"))
    ap.add_argument("--calib-dir")
    ap.add_argument("--calib-max", type=int, default=200)
    ap.add_argument("--calib-random", type=int, default=8)
    ap.add_argument("--percentile", type=float, default=99.99,
                    help="activation clipping percentile (100 = max)")
    ap.add_argument("--golden-image", help="image used for golden vectors")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    P = dict(np.load(args.weights))
    xs = load_calibration(args)
    scales = calibrate(P, xs, args.percentile)
    for n in ACT_NAMES:
        print(f"  scale[{n:6s}] = {scales[n]:.6g}")
    layers = quantize(P, scales)

    os.makedirs(os.path.join(args.out, "golden"), exist_ok=True)
    meta = dict(model="torchvision.alexnet IMAGENET1K_V1", input_scale=scales["input"],
                act_scales=scales, percentile=args.percentile, layers=[])
    gdir = os.path.join(args.out, "golden")
    with open(os.path.join(gdir, "layers.txt"), "w") as lf:
        for i, q in enumerate(layers):
            L = q["L"]
            np.save(os.path.join(args.out, f"layer{i}_weights.npy"), q["words"])
            np.save(os.path.join(args.out, f"layer{i}_params.npy"), q["params"])
            q["words"].tofile(os.path.join(gdir, f"w{i}.bin"))
            q["params"].tofile(os.path.join(gdir, f"p{i}.bin"))
            meta["layers"].append(dict(L, weight_words=int(q["words"].size),
                                       s_in=q["s_in"], s_out=q["s_out"],
                                       worst_case_acc=q["worst_acc"]))
            lf.write(" ".join(str(L[k]) for k in
                              ["op", "cin", "hin", "win", "cout", "k", "pad", "hout", "wout",
                               "pool", "relu", "raw"]) + f" {q['words'].size} {ac.out_words(L)}\n")
    fc8 = layers[-1]
    np.save(os.path.join(args.out, "fc8_logit_scale.npy"),
            (fc8["s_in"] * fc8["s_w"]).astype(np.float32))
    with open(os.path.join(args.out, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)

    # ---- golden vectors -------------------------------------------------
    if args.golden_image:
        from PIL import Image
        xg = ac.preprocess_pil(Image.open(args.golden_image))
    else:
        xg = xs[0]
    inp, outs = run_int(layers, xg, scales["input"])
    inp.tofile(os.path.join(gdir, "input.bin"))
    for i, o in enumerate(outs):
        o.tofile(os.path.join(gdir, f"golden{i}.bin"))
    np.save(os.path.join(args.out, "golden_input.npy"), inp)
    np.save(os.path.join(args.out, "golden_fc8_raw.npy"), outs[-1])

    lg_q = logits_from_raw(outs[-1], layers)
    lg_f = ac.alexnet_float(xg, P)
    print("golden image: float top1 =", int(np.argmax(lg_f)), " int8 top1 =", int(np.argmax(lg_q)))
    corr = np.corrcoef(lg_f, lg_q)[0, 1]
    print(f"golden image: float-vs-int8 logit correlation = {corr:.4f}")
    total = sum(q["words"].size for q in layers) * 8
    print(f"weight blobs: {total/2**20:.2f} MiB")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
