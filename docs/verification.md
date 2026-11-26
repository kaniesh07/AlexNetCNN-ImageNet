# 10. End-to-end verification and debugging guide

Everything below is bit-exact by construction: one integer contract (`pynq/alexnet_common.py`) is implemented by the Python golden model, the HLS C++ kernel, the ARM NumPy baseline, and (for its subset) the handwritten RTL. Any mismatch is a bug, not "quantisation noise".

## Verification status of this package

| Check | Tool | Status in this package |
|---|---|---|
| numpy fp32 AlexNet == torchvision | `tests/reference_test.py` | **ran** (random-init weights; max rel. err 6e-7). Re-run with the pretrained checkpoint on your host |
| preprocessing == torchvision transforms | `tests/reference_test.py` | **ran**, identical (0 difference) |
| conv1 space-to-depth exactness, packing, requant, multiplier, fc6 order | `tests/quantization_test.py` | **ran**, pass |
| HLS kernel unit tests vs naive C++ reference | g++ + AMD open-source `ap_int` | **ran**, pass |
| HLS kernel full 8-layer AlexNet vs Python golden | g++ csim | **ran**, all 8 layers bit-exact (random-init weights) |
| Handwritten RTL units + stream top + HLS-golden pool packet | Icarus Verilog 12 | **ran**, pass (mutation check confirmed the TB detects errors) |
| PYNQ driver flow vs behavioural IP model | `tests/driver_mock_test.py` | **ran**, pass |
| Vitis HLS csim / **csynth** / **cosim** / export | Vitis HLS 2024.2 | **not run** (no AMD tools in the build sandbox) |
| Vivado block design / synthesis / timing / bitstream | Vivado 2024.2 | **not run** |
| On-board Stage D/E | PYNQ-Z2 | **not run** |
| Accuracy with the real pretrained weights | ImageNet val | **not run** (checkpoint server unreachable from the sandbox) |

## Stage A — software reference
`python model/export_weights.py` then `python tests/reference_test.py --images <dir>`. Expected: forward error < 1e-4, preprocessing difference ≈ 0, sensible labels (e.g. a cat photo → "tabby"/"tiger cat"/"Egyptian cat" in top-5). If labels look random, the checkpoint or class file is wrong.

## Stage B — HLS kernel
1. `make -f Makefile.gcc run-full` (seconds) or `csim` in Vitis HLS. The unit tests cover: K = 3 and 5; pad 0/1/2; pool on/off; ReLU on/off; non-square maps; 1–3 channel groups; saturation (|weights| = 127, large inputs); negative values; FC requant and raw int32; all status-code paths; output-overrun sentinels.
2. Full network: `ALEXNET_GOLDEN=.../build/deploy/golden` → every layer word-identical.
3. `cosim` on the unit layers (verifies the generated RTL + AXI masters against the same TB).
Overflow analysis: dot8 needs 18 bits (8·128·128 = 2¹⁷), accumulator int32 with per-layer worst-case bound asserted at export, requant product < 2⁴⁷ in a 48-bit signed.

## Stage C — RTL
`rtl/testbench/run_xsim.sh`: leaf-unit tests (2000 MAC sequences incl. −128·−128, exhaustive ReLU, 500 pool windows) and the AXIS top with random TVALID gaps and random TREADY back-pressure, AXIS stability assertions on both ports, TLAST checks, packet counting, watchdog. One packet uses **real conv5 activations** reconstructed from the HLS golden data and checks the RTL max-pool output against the HLS pool5 output (`rtl_stream_model.golden_pool_packet`) — the HLS ↔ handwritten-RTL comparison. The RTL conv window path is checked against the same `requant` formula as the HLS kernel with M = 1.

## Stage D — FPGA
1. `sudo -E python3 rtl_stream_test.py` → DMA status registers error-free, all packets PASS.
2. `sudo -E python3 alexnet_inference.py --verify-golden` → register self-test, then 8 layers, each compared word-for-word with `deploy/golden/golden{i}.bin`.
3. Full inference on images; `--compare-arm` confirms FPGA fc8 == ARM NumPy fc8 bit-exactly.

## Debugging guide
| Symptom | Likely cause → action |
|---|---|
| `Overlay(...)` fails to parse / IP missing | `.bit`/`.hwh` names differ or come from different builds → copy both from `build/overlay` |
| `register map mismatch between xalexnet_accel_hw.h and .hwh` | header from another HLS export → rebuild or copy the matching header |
| `AcceleratorError: timeout` with AP_CTRL = 0x1 (start, not idle) | HP ports not enabled/clocked, or reset held → check `S_AXI_HPx_ACLK`, `peripheral_aresetn`; check that addresses written are `physical_address` values |
| kernel status 2/3/4 | layer table vs build constants mismatch (e.g. edited `TO`) → `alexnet_common.py` and `alexnet_accel.h` must agree |
| L0 mismatches, later layers garbage | input packing / scale wrong → compare `golden_input.npy` path first (`--verify-golden` uses the stored input) |
| sporadic mismatches that change run to run | cache coherency: buffers are non-cacheable by default in PYNQ; if you allocate `cacheable=True`, keep the `flush()`/`invalidate()` calls |
| `allocate` raises / "Failed to allocate memory" | CMA exhausted → reboot, check `CmaFree`, enlarge `cma=` |
| DMA test hangs, S2MM never idle | packet without a complete window/TLAST (the RTL emits TLAST only with a result), or S2MM length shorter than output → use `rtl_stream_model` builders |
| DMA `DMASlvErr`/`DMADecErr` | DMA master not mapped to DDR in the address editor (HP2 segment excluded) |
| **Timing (WNS < 0)** | look at the worst path in `timing_summary.rpt`. In the MAC loop: increase pipeline depth of requant (it is already outside the recurrence), or add `#pragma HLS BIND_OP ... latency=` on multipliers; last resort: 80 MHz (`create_clock -period 12.5` in HLS **and** `PCW_FPGA0_PERIPHERAL_FREQMHZ 80`) |
| DSP over 220 | `#pragma HLS BIND_OP variable=p op=mul impl=fabric` in `dot8` for half the lanes, or reduce `TO` to 8 (update `alexnet_common.TO`, re-export blobs) |
| Top-1 much worse than fp32 | calibration set too small / not representative; try percentile 99.999; check `scale[...]` printouts are not tiny (dead layer) |
