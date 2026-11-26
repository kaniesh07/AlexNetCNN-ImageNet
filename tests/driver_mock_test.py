#!/usr/bin/env python3
"""
driver_mock_test.py -- runs the REAL board driver (pynq/accelerator_driver.py,
hw_header.py) against tests/mock_pynq (behavioural model of the IP) and the
deploy blobs from model/quantize_model.py.  Checks: header parsing, register
programming of all 8 layers, ping-pong buffering, ap_done/ap_return handling,
golden comparison and buffer cleanup.  Needs build/deploy (run the model flow).
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pynq"))
sys.path.insert(0, os.path.join(ROOT, "tests", "mock_pynq"))
import pynq                                   # noqa: E402  (the mock)
import alexnet_common as ac                   # noqa: E402
import hw_header                              # noqa: E402
from accelerator_driver import AlexNetAccel   # noqa: E402

DEPLOY = os.path.join(ROOT, "build", "deploy")
FAKE_HDR = os.path.join(ROOT, "build", "fake_xalexnet_accel_hw.h")


def write_fake_header():
    # same #define style as Vitis HLS output; offsets are arbitrary on purpose
    names = ["act_in", "act_out", "weights", "params", "op", "cin", "hin", "win", "cout",
             "ksize", "pad", "hout", "wout", "pool", "relu", "raw_out"]
    lines = ["#define XALEXNET_ACCEL_CONTROL_ADDR_AP_CTRL 0x00",
             "#define XALEXNET_ACCEL_CONTROL_ADDR_AP_RETURN 0x10",
             "#define XALEXNET_ACCEL_CONTROL_BITS_AP_RETURN 32"]
    off = 0x18
    for n in names:
        lines.append(f"#define XALEXNET_ACCEL_CONTROL_ADDR_{n.upper()}_DATA 0x{off:x}")
        lines.append(f"#define XALEXNET_ACCEL_CONTROL_BITS_{n.upper()}_DATA 32")
        off += 8
    open(FAKE_HDR, "w").write("\n".join(lines) + "\n")
    rm = hw_header.parse_hls_header(FAKE_HDR)
    pynq.REGS.update({"AP_CTRL": rm.ap_ctrl, "AP_RETURN": rm.ap_return})
    pynq.REGS.update({n.upper() + "_DATA": rm.args[n][0] for n in names})


def main():
    write_fake_header()
    ol = pynq.Overlay("alexnet.bit")
    with AlexNetAccel(ol, DEPLOY, FAKE_HDR) as acc:
        assert acc.self_test_registers()
        inp = np.load(os.path.join(DEPLOY, "golden_input.npy"))
        raw, timing, inter = acc.infer(inp, keep_intermediates=True)
        for i, got in enumerate(inter):
            exp = np.fromfile(os.path.join(DEPLOY, "golden", f"golden{i}.bin"), dtype=np.uint64)
            assert np.array_equal(got, exp[: got.size]), f"layer {i} mismatch"
        exp_raw = ac.unpack_raw_i32(np.load(os.path.join(DEPLOY, "golden_fc8_raw.npy")), 1000)
        assert np.array_equal(raw, exp_raw)
        nbuf = len(acc._bufs)
    assert len(pynq._MEM) == 0, "buffers not freed"
    print(f"driver mock test: 8 layers bit-exact, {nbuf} buffers allocated and freed -> PASS")


if __name__ == "__main__":
    main()
