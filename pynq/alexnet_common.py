"""
alexnet_common.py -- single source of truth for the numerical contract shared by
the host tools (model/, tests/) and the PYNQ board code (pynq/).

Depends ONLY on numpy so it runs unchanged on the PYNQ-Z2 (Cortex-A9) and the
host PC.  Everything that the HLS kernel does is mirrored here bit-exactly:

  * activation memory layout "C8HW" (8 int8 channels packed per 64-bit word)
  * conv1 space-to-depth rewrite (11x11/s4 -> 3x3/s1 on 48 channels)
  * conv / FC weight packing in the exact order the kernel streams them
  * per-channel params word: bias(int32) | M(uint16) | shift(6b)
  * requantisation: out = sat8((acc*M + 2^(sh-1)) >> sh), optional ReLU
  * 3x3 / stride-2 max pooling

Model: torchvision.models.alexnet, weights AlexNet_Weights.IMAGENET1K_V1
(input 3x224x224; NOT the original 227x227 Krizhevsky network, no LRN).
"""
import numpy as np

# --------------------------------------------------------------------------
# Kernel compile-time constants (must equal hls/include/alexnet_accel.h)
# --------------------------------------------------------------------------
TI = 8            # input channels per 64-bit word
TO = 16           # output channels computed in parallel per conv tile
IN_BUF_WORDS = 19494   # 48/8 * 57 * 57  (largest conv input, conv1')
W_DEPTH = 432          # max (Cin/8)*K*K   (conv4: 48*9)
OUT_TILE_DEPTH = 3025  # max Hout*Wout     (conv1': 55*55)
POOL_DEPTH = 729       # max pooled pixels (27*27)
P_DEPTH = 4096         # max output channels / neurons per layer

OP_CONV, OP_FC = 0, 1

# torchvision ImageNet preprocessing (AlexNet_Weights.IMAGENET1K_V1.transforms)
RESIZE = 256
CROP = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# --------------------------------------------------------------------------
# Hardware layer table.  conv1 is executed as conv1' (space-to-depth form).
# hout/wout are the conv outputs BEFORE pooling.
# --------------------------------------------------------------------------
HW_LAYERS = [
    # name    op       cin   hin win cout  k pad hout wout pool relu raw
    dict(name="conv1", op=OP_CONV, cin=48,   hin=57, win=57, cout=64,   k=3, pad=0, hout=55, wout=55, pool=1, relu=1, raw=0),
    dict(name="conv2", op=OP_CONV, cin=64,   hin=27, win=27, cout=192,  k=5, pad=2, hout=27, wout=27, pool=1, relu=1, raw=0),
    dict(name="conv3", op=OP_CONV, cin=192,  hin=13, win=13, cout=384,  k=3, pad=1, hout=13, wout=13, pool=0, relu=1, raw=0),
    dict(name="conv4", op=OP_CONV, cin=384,  hin=13, win=13, cout=256,  k=3, pad=1, hout=13, wout=13, pool=0, relu=1, raw=0),
    dict(name="conv5", op=OP_CONV, cin=256,  hin=13, win=13, cout=256,  k=3, pad=1, hout=13, wout=13, pool=1, relu=1, raw=0),
    dict(name="fc6",   op=OP_FC,   cin=9216, hin=1,  win=1,  cout=4096, k=1, pad=0, hout=1,  wout=1,  pool=0, relu=1, raw=0),
    dict(name="fc7",   op=OP_FC,   cin=4096, hin=1,  win=1,  cout=4096, k=1, pad=0, hout=1,  wout=1,  pool=0, relu=1, raw=0),
    dict(name="fc8",   op=OP_FC,   cin=4096, hin=1,  win=1,  cout=1000, k=1, pad=0, hout=1,  wout=1,  pool=0, relu=0, raw=1),
]

# torchvision state_dict keys for each layer
TORCH_KEYS = {
    "conv1": "features.0", "conv2": "features.3", "conv3": "features.6",
    "conv4": "features.8", "conv5": "features.10",
    "fc6": "classifier.1", "fc7": "classifier.4", "fc8": "classifier.6",
}


def pooled_dims(L):
    if L["pool"]:
        return (L["hout"] - 3) // 2 + 1, (L["wout"] - 3) // 2 + 1
    return L["hout"], L["wout"]


def out_words(L):
    """Number of 64-bit words the kernel writes for layer L."""
    if L["op"] == OP_FC:
        return L["cout"] // 2 if L["raw"] else L["cout"] // 8
    hp, wp = pooled_dims(L)
    return (L["cout"] // 8) * hp * wp


def in_words(L):
    return (L["cin"] // 8) * L["hin"] * L["win"]


# --------------------------------------------------------------------------
# Preprocessing (PIL-based; identical to torchvision on PIL images)
# --------------------------------------------------------------------------
def preprocess_pil(img):
    """PIL.Image -> float32 CHW normalised tensor (3,224,224)."""
    from PIL import Image
    img = img.convert("RGB")
    w, h = img.size
    if w <= h:
        nw, nh = RESIZE, int(RESIZE * h / w)
    else:
        nh, nw = RESIZE, int(RESIZE * w / h)
    img = img.resize((nw, nh), Image.BILINEAR)
    left = int(round((nw - CROP) / 2.0))
    top = int(round((nh - CROP) / 2.0))
    img = img.crop((left, top, left + CROP, top + CROP))
    x = np.asarray(img, dtype=np.float32) / 255.0          # HWC
    x = (x - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))       # CHW


# --------------------------------------------------------------------------
# Quantisation primitives
# --------------------------------------------------------------------------
def quantize_tensor(x, scale):
    """float -> int8, round-half-to-even (numpy rint), saturating."""
    return np.clip(np.rint(x / scale), -128, 127).astype(np.int8)


def choose_multiplier(m):
    """Real multiplier m>0 -> (M, sh) with M in [2^14, 2^15) where possible,
    sh in [1, 47].  out = (acc*M + 2^(sh-1)) >> sh  ~= acc*m."""
    m = np.asarray(m, dtype=np.float64)
    M = np.zeros(m.shape, dtype=np.int64)
    sh = np.zeros(m.shape, dtype=np.int64)
    for i, mi in np.ndenumerate(m):
        if not np.isfinite(mi) or mi <= 0:
            M[i], sh[i] = 0, 1
            continue
        s = 14 - int(np.floor(np.log2(mi)))
        s = min(max(s, 1), 47)
        q = int(round(mi * (1 << s)))
        if q >= 32768:          # rounding overflow or m too large for sh>=1
            if s > 1:
                s -= 1
                q = int(round(mi * (1 << s)))
            q = min(q, 32767)
        M[i], sh[i] = q, s
    return M, sh


def requant(acc, M, sh, relu):
    """Bit-exact model of the kernel's requant(): acc int (any shape, channel
    axis 0), M/sh per channel (len C).  Returns int8."""
    acc = acc.astype(np.int64)
    bshape = (-1,) + (1,) * (acc.ndim - 1)
    Mv = M.astype(np.int64).reshape(bshape)
    sv = np.maximum(sh.astype(np.int64), 1).reshape(bshape)   # kernel treats sh=0 as 1
    p = (acc * Mv + (np.int64(1) << (sv - 1))) >> sv
    p = np.clip(p, -128, 127)
    if relu:
        p = np.maximum(p, 0)
    return p.astype(np.int8)


def pack_params(bias_i32, M, sh):
    b = bias_i32.astype(np.int64) & 0xFFFFFFFF
    w = b | (M.astype(np.int64) << 32) | (sh.astype(np.int64) << 48)
    return w.astype(np.uint64)


# --------------------------------------------------------------------------
# Layout helpers
# --------------------------------------------------------------------------
def pack_c8hw(x):
    """int8 (C,H,W) -> uint64 words, layout [C/8][H][W], lane = c%8."""
    C, H, W = x.shape
    assert C % 8 == 0
    v = x.astype(np.int8).reshape(C // 8, 8, H, W).transpose(0, 2, 3, 1)
    return np.ascontiguousarray(v).view(np.uint64).reshape(-1)


def unpack_c8hw(words, C, H, W):
    v = np.ascontiguousarray(words[: (C // 8) * H * W]).view(np.int8)
    return v.reshape(C // 8, H, W, 8).transpose(0, 3, 1, 2).reshape(C, H, W)


def unpack_raw_i32(words, n):
    return np.ascontiguousarray(words[: n // 2]).view(np.int32)[:n].copy()


def space_to_depth_input(xq_pad):
    """int8 (3,228,228) zero-padded input -> (48,57,57); c' = c*16+dy*4+dx."""
    C, H, W = xq_pad.shape
    v = xq_pad.reshape(C, H // 4, 4, W // 4, 4)            # c Y dy X dx
    return np.ascontiguousarray(v.transpose(0, 2, 4, 1, 3).reshape(C * 16, H // 4, W // 4))


def space_to_depth_conv1_weights(w):
    """(64,3,11,11) -> (64,48,3,3) for the stride-1 equivalent conv."""
    O, C, K, _ = w.shape
    wp = np.zeros((O, C, 12, 12), dtype=w.dtype)
    wp[:, :, :K, :K] = w
    v = wp.reshape(O, C, 3, 4, 3, 4)                         # o c ky dy kx dx
    return np.ascontiguousarray(v.transpose(0, 1, 3, 5, 2, 4).reshape(O, C * 16, 3, 3))


def prepare_input(x_float, s_in):
    """normalised float (3,224,224) -> packed uint64 kernel input for conv1'."""
    xq = quantize_tensor(x_float, s_in)
    xp = np.zeros((3, 228, 228), dtype=np.int8)
    xp[:, 2:226, 2:226] = xq
    return pack_c8hw(space_to_depth_input(xp))


def pack_conv_weights(wq):
    """int8 (Cout,Cin,K,K) -> words in kernel streaming order
    [tile][to][cin/8][ky][kx] (8 input channels per word)."""
    O, C, K, _ = wq.shape
    assert O % TO == 0 and C % 8 == 0
    v = wq.reshape(O, C // 8, 8, K, K).transpose(0, 1, 3, 4, 2)   # o cig ky kx lane
    return np.ascontiguousarray(v.astype(np.int8)).view(np.uint64).reshape(-1)


def unpack_conv_weights(words, O, C, K):
    """Inverse of pack_conv_weights -> int8 (O, C, K, K)."""
    v = np.ascontiguousarray(words).view(np.int8).reshape(O, C // 8, K, K, 8)
    return np.ascontiguousarray(v.transpose(0, 1, 4, 2, 3).reshape(O, C, K, K))


def unpack_params(words):
    """uint64 params -> (bias int64, M int64, sh int64)."""
    w = words.astype(np.uint64)
    bias = (w & np.uint64(0xFFFFFFFF)).astype(np.int64)
    bias = np.where(bias >= 2 ** 31, bias - 2 ** 32, bias)
    M = ((w >> np.uint64(32)) & np.uint64(0xFFFF)).astype(np.int64)
    sh = ((w >> np.uint64(48)) & np.uint64(0x3F)).astype(np.int64)
    return bias, M, sh


def fc_input_permutation(cin, c_prev=None, h_prev=1, w_prev=1):
    """Index map: position in the kernel's packed input vector (word j, lane l
    -> flat j*8+l) -> torch flatten index (c*H*W + y*W + x)."""
    if c_prev is None:
        return np.arange(cin)
    hw = h_prev * w_prev
    j = np.arange(cin) // 8
    l = np.arange(cin) % 8
    c = (j // hw) * 8 + l
    pix = j % hw
    return c * hw + pix


def pack_fc_weights(wq, perm):
    """int8 (Cout,Cin) torch order -> words, row-major per neuron, with the
    input columns reordered to the kernel's C8HW flatten order."""
    w = wq[:, perm]
    return np.ascontiguousarray(w.astype(np.int8)).view(np.uint64).reshape(-1)


# --------------------------------------------------------------------------
# Reference implementations
# --------------------------------------------------------------------------
def _im2col(x, k, pad, stride=1):
    C, H, W = x.shape
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    Ho = (H + 2 * pad - k) // stride + 1
    Wo = (W + 2 * pad - k) // stride + 1
    s0, s1, s2 = xp.strides
    cols = np.lib.stride_tricks.as_strided(
        xp, shape=(C, k, k, Ho, Wo),
        strides=(s0, s1, s2, s1 * stride, s2 * stride))
    return cols.reshape(C * k * k, Ho * Wo), Ho, Wo


def maxpool3s2(x):
    C, H, W = x.shape
    Ho, Wo = (H - 3) // 2 + 1, (W - 3) // 2 + 1
    out = None
    for dy in range(3):
        for dx in range(3):
            v = x[:, dy:dy + 2 * Ho:2, dx:dx + 2 * Wo:2]
            out = v.copy() if out is None else np.maximum(out, v)
    return out


def conv_float(x, w, b, stride, pad):
    cols, Ho, Wo = _im2col(x.astype(np.float32), w.shape[2], pad, stride)
    y = w.reshape(w.shape[0], -1).astype(np.float32) @ cols + b[:, None]
    return y.reshape(w.shape[0], Ho, Wo)


def alexnet_float(x, P, record=None):
    """Float32 numpy forward pass of torchvision AlexNet.  P = dict of numpy
    weights keyed by torchvision state_dict names.  record(name, tensor) is
    called on every post-ReLU activation (for calibration)."""
    rec = record or (lambda n, t: None)
    cfg = [("features.0", 4, 2, True), ("features.3", 1, 2, True),
           ("features.6", 1, 1, False), ("features.8", 1, 1, False),
           ("features.10", 1, 1, True)]
    names = ["conv1", "conv2", "conv3", "conv4", "conv5"]
    for (key, s, p, pool), nm in zip(cfg, names):
        x = np.maximum(conv_float(x, P[key + ".weight"], P[key + ".bias"], s, p), 0)
        rec(nm, x)
        if pool:
            x = maxpool3s2(x)
    # AdaptiveAvgPool2d((6,6)) is the identity for a 6x6 map (224 input).
    assert x.shape == (256, 6, 6)
    v = x.reshape(-1)
    for key, nm, relu in [("classifier.1", "fc6", True), ("classifier.4", "fc7", True),
                          ("classifier.6", "fc8", False)]:
        v = P[key + ".weight"].astype(np.float32) @ v + P[key + ".bias"]
        if relu:
            v = np.maximum(v, 0)
            rec(nm, v)
    return v


def conv_int_acc(xq, wq, bias, pad):
    """Exact integer conv (stride 1).  float64 BLAS is exact here because all
    partial sums stay far below 2^53."""
    cols, Ho, Wo = _im2col(xq.astype(np.float64), wq.shape[2], pad, 1)
    acc = wq.reshape(wq.shape[0], -1).astype(np.float64) @ cols
    acc = acc.reshape(wq.shape[0], Ho, Wo) + bias.astype(np.float64)[:, None, None]
    return np.rint(acc).astype(np.int64)


def run_layer_int(L, x_words, wq, bias, M, sh, perm=None):
    """Bit-exact software model of ONE kernel invocation working on packed
    words, exactly like the FPGA.  wq: conv (Cout,Cin,K,K) int8 already in
    hardware form (conv1 = space-to-depth form); FC (Cout,Cin) torch order."""
    if L["op"] == OP_CONV:
        x = unpack_c8hw(x_words, L["cin"], L["hin"], L["win"])
        acc = conv_int_acc(x, wq, bias, L["pad"])
        y = requant(acc, M, sh, L["relu"])
        if L["pool"]:
            y = maxpool3s2(y)
        return pack_c8hw(y)
    v = np.ascontiguousarray(x_words[: L["cin"] // 8]).view(np.int8).astype(np.float64)
    w = wq[:, perm].astype(np.float64) if perm is not None else wq.astype(np.float64)
    acc = np.rint(w @ v).astype(np.int64) + bias.astype(np.int64)
    if L["raw"]:
        a = acc.astype(np.int32)
        return np.ascontiguousarray(a).view(np.uint64).copy()
    y = requant(acc, M, sh, L["relu"])
    return np.ascontiguousarray(y).view(np.uint64).copy()


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()
