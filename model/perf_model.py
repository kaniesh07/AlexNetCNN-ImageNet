#!/usr/bin/env python3
"""
perf_model.py -- ANALYTIC estimate (category: "estimated") of the AlexNet layer
engine's cycles, DDR traffic and on-chip memory, derived from the loop trip
counts in hls/src/*.cpp.  These are NOT synthesis or measurement results;
replace them with csynth / board numbers as soon as you have them.

Assumptions (edit below):
  F_CLK            100 MHz (PS FCLK_CLK0)
  HP_EFF_BYTES_CYC effective sustained bytes/cycle per HP port (64-bit port
                   peak = 8 B/cycle; 4-6 B/cycle is a typical observed range)
  LOOP_OVH         cycles of pipeline fill/drain + loop entry per loop call
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pynq"))
import alexnet_common as ac  # noqa: E402

F_CLK = 100e6
LOOP_OVH = 30


def layer_cycles(L, hp_eff):
    if L["op"] == ac.OP_CONV:
        cig = L["cin"] // 8
        kk = L["k"] ** 2
        cigkk = cig * kk
        tiles = L["cout"] // ac.TO
        hp, wp = ac.pooled_dims(L)
        npix = hp * wp
        mac = L["hout"] * L["wout"] * cigkk
        per_tile = (ac.TO * cigkk + LOOP_OVH) + (ac.TO + 5) + (mac + LOOP_OVH) \
            + ((hp * wp * 9 + LOOP_OVH) if L["pool"] else 0) + 2 * (npix + LOOP_OVH)
        cyc = ac.in_words(L) + L["cout"] + 3 * LOOP_OVH + tiles * per_tile
        bytes_rd = 8 * (ac.in_words(L) + L["cout"] + tiles * ac.TO * cigkk)
        macs = L["cout"] * L["cin"] * kk * L["hout"] * L["wout"]
        bound = "compute"
    else:
        cinw = L["cin"] // 8
        stream = L["cout"] * cinw
        cyc_compute = cinw + L["cout"] + stream + 3 * LOOP_OVH
        cyc_mem = 8 * stream / hp_eff            # weight stream limited by one HP port
        cyc = max(cyc_compute, cyc_mem)
        bound = "memory" if cyc_mem > cyc_compute else "compute"
        bytes_rd = 8 * (cinw + L["cout"] + stream)
        macs = L["cout"] * L["cin"]
    return cyc, bytes_rd, macs, bound


def main(hp_eff=5.0):
    print(f"ESTIMATE @ {F_CLK/1e6:.0f} MHz, HP effective {hp_eff} B/cycle "
          f"({hp_eff*F_CLK/1e6:.0f} MB/s), TO={ac.TO} x TI={ac.TI} = {ac.TO*ac.TI} MAC/cycle")
    print(f"{'layer':6s} {'MACs(M)':>9s} {'DDR rd(MB)':>10s} {'cycles(M)':>10s} {'ms':>8s}  bound")
    tot_c = tot_b = tot_m = 0
    for L in ac.HW_LAYERS:
        c, b, m, bound = layer_cycles(L, hp_eff)
        tot_c += c
        tot_b += b
        tot_m += m
        print(f"{L['name']:6s} {m/1e6:9.1f} {b/1e6:10.2f} {c/1e6:10.3f} {c/F_CLK*1e3:8.2f}  {bound}")
    print(f"{'total':6s} {tot_m/1e6:9.1f} {tot_b/1e6:10.2f} {tot_c/1e6:10.3f} {tot_c/F_CLK*1e3:8.2f}")
    print(f"(conv1 MACs include the 12x12 zero-padded space-to-depth kernel: "
          f"{(12/11)**2:.3f}x the original 11x11)")


def bram_estimate():
    # 36Kb BRAM in 512x72 mode; 18Kb = half
    rows = [
        ("in_buf   19494 x 64b", 19494, 64),
        ("w_buf    16 banks x 432 x 64b", 432, 64, 16),
        ("out_tile 3025 x 128b", 3025, 128),
        ("pooled   729 x 128b", 729, 128),
        ("p_buf    4096 x 64b", 4096, 64),
    ]
    print("\nBRAM36 estimate (512x72 primitive packing; HLS may differ):")
    total = 0
    for r in rows:
        name, depth, width = r[0], r[1], r[2]
        banks = r[3] if len(r) > 3 else 1
        per = -(-depth // 512) * -(-width // 72)
        n = per * banks
        total += n
        print(f"  {name:32s} ~{n:4d}")
    print(f"  {'buffers total':32s} ~{total:4d}  of 140 (+ m_axi FIFOs, DMA, interconnect)")


if __name__ == "__main__":
    for eff in (4.0, 6.0):
        main(eff)
        print()
    bram_estimate()
