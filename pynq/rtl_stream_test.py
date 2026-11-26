#!/usr/bin/env python3
"""
rtl_stream_test.py -- on-board test of the handwritten RTL (rtl_stream_top)
through AXI DMA (simple/direct-register mode, no scatter-gather).

For every packet type from rtl_stream_model.py: MM2S sends the packet,
S2MM receives the results, both channels are polled with a timeout, the DMA
status registers are checked for errors, and the data is compared with the
golden model.  DMA register offsets used below are fixed by the AXI DMA IP
(PG021: MM2S_DMACR 0x00, MM2S_DMASR 0x04, S2MM_DMACR 0x30, S2MM_DMASR 0x34).

Usage:  sudo python3 rtl_stream_test.py [--overlay overlay/alexnet.bit]
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rtl_stream_model as m                       # noqa: E402
from accelerator_driver import find_ip             # noqa: E402

MM2S_DMASR, S2MM_DMASR = 0x04, 0x34
SR_HALTED, SR_IDLE = 0x1, 0x2
SR_ERRS = {0x10: "DMAIntErr", 0x20: "DMASlvErr", 0x40: "DMADecErr"}


def dma_errors(dma):
    out = []
    for name, off in (("MM2S", MM2S_DMASR), ("S2MM", S2MM_DMASR)):
        sr = dma.read(off)
        out += [f"{name}:{t}" for bit, t in SR_ERRS.items() if sr & bit]
    return out


def wait_idle(dma, off, timeout):
    t0 = time.perf_counter()
    while True:
        sr = dma.read(off)
        if sr & (SR_IDLE | SR_HALTED):
            return time.perf_counter() - t0
        if time.perf_counter() - t0 > timeout:
            raise TimeoutError(f"DMA SR@0x{off:02x}=0x{sr:08x} not idle after {timeout}s")


def run_packet(dma, beats_in, beats_out, timeout):
    from pynq import allocate
    tx = allocate(shape=(len(beats_in),), dtype=np.uint32)
    rx = allocate(shape=(len(beats_out),), dtype=np.uint32)
    try:
        tx[:] = [d for d, _ in beats_in]
        rx[:] = 0
        tx.flush()
        rx.flush()
        dma.recvchannel.transfer(rx)        # arm S2MM first
        dma.sendchannel.transfer(tx)
        t_tx = wait_idle(dma, MM2S_DMASR, timeout)
        t_rx = wait_idle(dma, S2MM_DMASR, timeout)
        errs = dma_errors(dma)
        rx.invalidate()
        got = np.array(rx, dtype=np.uint32)
        exp = np.array([d for d, _ in beats_out], dtype=np.uint32)
        return got, exp, errs, max(t_tx, t_rx)
    finally:
        tx.freebuffer()
        rx.freebuffer()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlay", default=os.path.join(HERE, "overlay", "alexnet.bit"))
    ap.add_argument("--deploy", default=os.path.join(HERE, "deploy"))
    ap.add_argument("--timeout", type=float, default=1.0)
    a = ap.parse_args()
    from pynq import Overlay
    ol = Overlay(a.overlay)
    dma = getattr(ol, find_ip(ol, "xilinx.com:ip:axi_dma").replace("/", "."))
    rng = np.random.default_rng(3)
    np.random.seed(3)
    ri = lambda *s: rng.integers(-128, 128, size=s).astype(np.int8)  # noqa: E731
    tests = [
        ("relu", m.relu_packet(ri(256, 4), 1)),
        ("passthrough", m.relu_packet(ri(256, 4), 0)),
        ("maxpool", m.pool_packet(ri(64, 9, 4), 0)),
        ("conv-raw", m.conv_packet(ri(9), -1000, ri(64, 9), 0, 0, 0)),
        ("conv-requant", m.conv_packet(ri(9), 300, ri(64, 9), 8, 1, 1)),
    ]
    if os.path.isdir(os.path.join(a.deploy, "golden")):
        tests.append(("hls-golden-pool", m.golden_pool_packet(a.deploy, 4, 0)))
    fails = 0
    for name, (bi, bo) in tests:
        got, exp, errs, dt = run_packet(dma, bi, bo, a.timeout)
        bad = int(np.count_nonzero(got != exp))
        ok = bad == 0 and not errs
        fails += not ok
        print(f"{name:16s} in={len(bi):5d} out={len(bo):5d} mismatches={bad:4d} "
              f"dma_err={errs or '-'} t={dt*1e6:8.1f} us  {'PASS' if ok else 'FAIL'}")
    print("RTL STREAM TEST:", "PASS" if fails == 0 else f"FAIL ({fails})")


if __name__ == "__main__":
    main()
