#!/usr/bin/env python3
"""
quantization_test.py -- numerical-contract tests (HOST, numpy only).

Q1  conv1 space-to-depth rewrite is EXACT (float and integer)
Q2  pack/unpack round trips (C8HW, conv weights, params)
Q3  requant(): rounding (half-up), saturation at -128/127, ReLU, sh clamp
Q4  choose_multiplier(): relative error < 2^-14 over 1e-6..2
Q5  fc6 column permutation matches torch.flatten order of pool5
Q6  int8 network vs float network on the calibration/eval images
    (logit correlation, top-1/top-5 agreement) -- needs build/ artefacts
Q7  ARM-only int8 model (pynq/arm_reference.py) is bit-exact with golden
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pynq"))
sys.path.insert(0, os.path.join(ROOT, "model"))
import alexnet_common as ac  # noqa: E402

rng = np.random.default_rng(42)


def q1_space_to_depth():
    x = rng.standard_normal((3, 224, 224)).astype(np.float32)
    w = rng.standard_normal((64, 3, 11, 11)).astype(np.float32)
    b = np.zeros(64, np.float32)
    ref = ac.conv_float(x, w, b, 4, 2)                         # 55x55
    xp = np.pad(x, ((0, 0), (2, 2), (2, 2)))
    s2d = ac.space_to_depth_input(xp)
    got = ac.conv_float(s2d, ac.space_to_depth_conv1_weights(w), b, 1, 0)
    assert got.shape == ref.shape == (64, 55, 55)
    assert np.allclose(ref, got, rtol=1e-4, atol=1e-3), np.abs(ref - got).max()
    xi = rng.integers(-128, 128, (3, 228, 228)).astype(np.int8)
    wi = rng.integers(-127, 128, (64, 3, 11, 11)).astype(np.int8)
    bi = rng.integers(-1000, 1000, 64)
    ref_i = np.rint(ac.conv_float(xi.astype(np.float64), wi.astype(np.float64),
                                  bi.astype(np.float64), 4, 0)).astype(np.int64)
    got_i = ac.conv_int_acc(ac.space_to_depth_input(xi), ac.space_to_depth_conv1_weights(wi), bi, 0)
    assert np.array_equal(ref_i, got_i)
    print("[Q1] conv1 11x11/s4 == space-to-depth 3x3/s1 (float and integer exact)  PASS")


def q2_packing():
    x = rng.integers(-128, 128, (64, 13, 11)).astype(np.int8)
    assert np.array_equal(ac.unpack_c8hw(ac.pack_c8hw(x), 64, 13, 11), x)
    w = rng.integers(-127, 128, (32, 16, 5, 5)).astype(np.int8)
    assert np.array_equal(ac.unpack_conv_weights(ac.pack_conv_weights(w), 32, 16, 5), w)
    b = rng.integers(-2 ** 31, 2 ** 31, 100)
    M = rng.integers(0, 32768, 100)
    sh = rng.integers(1, 48, 100)
    b2, M2, s2 = ac.unpack_params(ac.pack_params(b, M, sh))
    assert np.array_equal(b, b2) and np.array_equal(M, M2) and np.array_equal(sh, s2)
    # lane order: channel c%8 in byte c%8 of the word
    t = np.zeros((8, 1, 1), np.int8)
    t[3, 0, 0] = 5
    assert int(ac.pack_c8hw(t)[0]) == 5 << 24
    print("[Q2] C8HW / weight / param pack-unpack round trips, lane order  PASS")


def q3_requant():
    M = np.array([1 << 14])
    sh = np.array([15])                                         # x0.5
    acc = np.array([[1], [2], [3], [-1], [-2], [-3], [255], [256], [10 ** 6], [-10 ** 6]])
    got = ac.requant(acc.T, M, sh, False)[0]
    # (acc*16384 + 16384) >> 15  == floor(acc/2 + 0.5)  (round half up)
    exp = np.clip(np.floor(acc[:, 0] / 2 + 0.5), -128, 127)
    assert np.array_equal(got, exp), (got, exp)
    assert np.array_equal(ac.requant(np.array([[-5, 5]]), np.array([1 << 14]), np.array([14]), True)[0], [0, 5])
    assert ac.requant(np.array([[3]]), np.array([1]), np.array([0]), False)[0, 0] == 2   # sh=0 -> 1
    print("[Q3] requant rounding / saturation / ReLU / shift clamp  PASS")


def q4_multiplier():
    m = np.exp(rng.uniform(np.log(1e-6), np.log(2.0), 5000))
    M, sh = ac.choose_multiplier(m)
    assert M.max() < 32768 and sh.min() >= 1 and sh.max() <= 47
    rel = np.abs(M / 2.0 ** sh - m) / m
    assert rel.max() < 2.0 ** -14, rel.max()
    print(f"[Q4] choose_multiplier max relative error {rel.max():.2e} (< 2^-14)  PASS")


def q5_fc6_perm():
    pool5 = rng.integers(-128, 128, (256, 6, 6)).astype(np.int8)
    torch_flat = pool5.reshape(-1)                              # torch.flatten order
    packed = ac.pack_c8hw(pool5).view(np.int8)                  # kernel order
    perm = ac.fc_input_permutation(9216, 256, 6, 6)
    assert np.array_equal(torch_flat[perm], packed)
    print("[Q5] fc6 input permutation == torch flatten of pool5  PASS")


def q6_q7_network():
    npz = os.path.join(ROOT, "build", "alexnet_fp32.npz")
    deploy = os.path.join(ROOT, "build", "deploy")
    if not (os.path.exists(npz) and os.path.isdir(deploy)):
        print("[Q6] skipped (run model/export_weights.py and model/quantize_model.py first)")
        return
    import json
    import quantize_model as qm
    P = dict(np.load(npz))
    meta = json.load(open(os.path.join(deploy, "model_meta.json")))
    scales = meta["act_scales"]
    layers = qm.quantize(P, scales)
    imgs_dir = os.environ.get("EVAL_IMAGES")
    xs = []
    if imgs_dir:
        import glob
        from PIL import Image
        for f in sorted(glob.glob(os.path.join(imgs_dir, "*")))[:50]:
            try:
                xs.append(ac.preprocess_pil(Image.open(f)))
            except Exception:
                pass
    if not xs:
        for _ in range(4):
            img = rng.random((3, 28, 28)).repeat(8, 1).repeat(8, 2)
            xs.append(((img.transpose(1, 2, 0) - ac.MEAN) / ac.STD).transpose(2, 0, 1).astype(np.float32))
    t1 = t5 = 0
    corr = []
    for x in xs:
        lf = ac.alexnet_float(x, P)
        _, outs = qm.run_int(layers, x, scales["input"])
        lq = qm.logits_from_raw(outs[-1], layers)
        corr.append(np.corrcoef(lf, lq)[0, 1])
        t1 += int(np.argmax(lf) == np.argmax(lq))
        t5 += int(np.argmax(lf) in np.argsort(-lq)[:5])
    print(f"[Q6] int8 vs fp32 on {len(xs)} images: top-1 agreement {t1}/{len(xs)}, "
          f"top-5 contains fp32 top-1 {t5}/{len(xs)}, mean logit corr {np.mean(corr):.4f}"
          + ("" if imgs_dir else "  (synthetic images; set EVAL_IMAGES=dir for real ones)"))
    from arm_reference import ArmInt8Model
    arm = ArmInt8Model(deploy)
    raw = arm.forward(np.load(os.path.join(deploy, "golden_input.npy")))
    exp = ac.unpack_raw_i32(np.load(os.path.join(deploy, "golden_fc8_raw.npy")), 1000)
    assert np.array_equal(raw, exp)
    print("[Q7] ARM-only int8 model bit-exact with golden fc8  PASS")


if __name__ == "__main__":
    q1_space_to_depth()
    q2_packing()
    q3_requant()
    q4_multiplier()
    q5_fc6_perm()
    q6_q7_network()
    print("quantization_test: ALL PASS")
