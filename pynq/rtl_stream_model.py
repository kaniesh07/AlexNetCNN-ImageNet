"""
rtl_stream_model.py -- packet builder + bit-exact golden model for the
handwritten RTL block rtl/rtl_stream_top.v.  numpy only: used by
rtl/testbench/gen_vectors.py (simulation) and pynq/rtl_stream_test.py (board).

A "beat" is (data:uint32, last:bool).
"""
import numpy as np

OP_RELU, OP_POOL, OP_CONV = 0, 1, 2


def header(op, relu=0, shift=0, requant=0):
    return (op & 0xFF) | ((relu & 1) << 8) | ((shift & 0x1F) << 16) | ((requant & 1) << 24)


def pack4(lanes):
    """4 int8 -> uint32 (lane 0 in bits 7:0)."""
    b = np.asarray(lanes, dtype=np.int8).astype(np.uint8).astype(np.uint32)
    return int(b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24))


def unpack4(word):
    return np.array([(word >> (8 * l)) & 0xFF for l in range(4)], dtype=np.uint8).astype(np.int8)


def _beats(words):
    return [(int(w) & 0xFFFFFFFF, i == len(words) - 1) for i, w in enumerate(words)]


def relu_packet(data_lanes, relu):
    """data_lanes: int8 (n,4).  Returns (input beats, expected output beats)."""
    data_lanes = np.asarray(data_lanes, dtype=np.int8)
    words = [header(OP_RELU, relu)] + [pack4(r) for r in data_lanes]
    y = np.maximum(data_lanes, 0) if relu else data_lanes
    return _beats(words), _beats([pack4(r) for r in y])


def pool_packet(windows, relu):
    """windows: int8 (n, 9, 4) = n windows x 9 positions x 4 channels."""
    windows = np.asarray(windows, dtype=np.int8)
    words = [header(OP_POOL, relu)]
    for w in windows:
        words += [pack4(r) for r in w]
    y = windows.max(axis=1)
    if relu:
        y = np.maximum(y, 0)
    return _beats(words), _beats([pack4(r) for r in y])


def conv_result(acc, shift, requant, relu):
    acc = int(acc)
    if requant:
        r = acc if shift == 0 else (acc + (1 << (shift - 1))) >> shift
        r = max(-128, min(127, r))
        if relu and r < 0:
            r = 0
        return r & 0xFFFFFFFF                    # sign-extended to 32 bits
    a = ((acc + 2 ** 31) % 2 ** 32) - 2 ** 31    # int32 wrap
    if relu and a < 0:
        a = 0
    return a & 0xFFFFFFFF


def conv_packet(w9, bias, windows, shift, requant, relu):
    """w9: int8 (9,), bias int32, windows int8 (n, 9)."""
    w9 = np.asarray(w9, dtype=np.int8)
    w12 = np.zeros(12, np.int8)
    w12[:9] = w9
    words = [header(OP_CONV, relu, shift, requant)]
    words += [pack4(w12[0:4]), pack4(w12[4:8]), pack4(w12[8:12]), int(bias) & 0xFFFFFFFF]
    out = []
    for win in np.asarray(windows, dtype=np.int8):
        x12 = np.zeros(12, np.int8)
        x12[:9] = win
        x12_noise = x12.copy()
        x12_noise[9:] = np.random.randint(-128, 128, 3)   # ignored lanes must not matter
        words += [pack4(x12_noise[0:4]), pack4(x12_noise[4:8]), pack4(x12_noise[8:12])]
        acc = int(bias) + int(np.dot(w9.astype(np.int64), win.astype(np.int64)))
        out.append(conv_result(acc, shift, requant, relu))
    return _beats(words), _beats(out)


def golden_pool_packet(deploy_dir, layer=4, ch0=0):
    """Cross-check against the HLS kernel: rebuild conv5's pre-pool int8
    activations from the HLS-verified golden data, gather the 3x3 windows
    of 4 channels, and expect the pooled values stored in golden{layer}.bin
    (which the HLS C simulation reproduces bit-exactly)."""
    import os
    import alexnet_common as ac
    L = ac.HW_LAYERS[layer]
    g = os.path.join(deploy_dir, "golden")
    x_words = np.fromfile(os.path.join(g, f"golden{layer-1}.bin"), dtype=np.uint64)
    w_words = np.load(os.path.join(deploy_dir, f"layer{layer}_weights.npy"))
    p_words = np.load(os.path.join(deploy_dir, f"layer{layer}_params.npy"))
    exp_words = np.fromfile(os.path.join(g, f"golden{layer}.bin"), dtype=np.uint64)
    wq = ac.unpack_conv_weights(w_words, L["cout"], L["cin"], L["k"])
    bias, M, sh = ac.unpack_params(p_words)
    x = ac.unpack_c8hw(x_words, L["cin"], L["hin"], L["win"])
    pre = ac.requant(ac.conv_int_acc(x, wq, bias, L["pad"]), M, sh, L["relu"])   # (C,13,13)
    hp, wp = ac.pooled_dims(L)
    expected = ac.unpack_c8hw(exp_words, L["cout"], hp, wp)                      # HLS output
    wins = []
    for py in range(hp):
        for px in range(wp):
            blk = pre[ch0:ch0 + 4, 2 * py:2 * py + 3, 2 * px:2 * px + 3]       # (4,3,3)
            wins.append(blk.reshape(4, 9).T)                                    # (9,4)
    beats_in, beats_out = pool_packet(np.array(wins), relu=0)
    hls = [pack4(expected[ch0:ch0 + 4, py, px]) for py in range(hp) for px in range(wp)]
    assert [b[0] for b in beats_out] == hls, "model pooled != HLS golden"
    return beats_in, beats_out
