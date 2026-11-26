# 11. Performance evaluation methodology

Report every number with its **category** — never mix them in one column:

| Category | Source | Example |
|---|---|---|
| Estimated | `model/perf_model.py` (analytic) | 154–203 ms accelerator time |
| Simulated | HLS csynth latency × trip counts, cosim cycle counts | cycles of `MAC` loop for a given layer |
| Synthesised | Vivado post-route `utilization.rpt`, `timing_summary.rpt`, `power.rpt` (vectorless estimate) | WNS, LUT/DSP/BRAM |
| Measured | board: `alexnet_inference.py`, `arm_reference.py`, external power meter | ms, images/s, W |

## Procedure
1. **Latency** — `sudo -E python3 alexnet_inference.py --image images/*.jpg --repeat 20`. Discard the first repetition (page faults / cold caches); report median and p95 of: preprocess (CPU), quantise+pack (CPU), each FPGA layer, accelerator total, end-to-end. Timer: `time.perf_counter()`; per-layer times include the PS register writes and polling granularity (µs-level).
2. **ARM-only baseline** — `python3 arm_reference.py --image ...` runs the identical int8 network in NumPy on the Cortex-A9 (bit-exact with the FPGA). Optionally also an fp32 NumPy run on the host for reference. Speed-up = ARM end-to-end / FPGA end-to-end.
3. **Agreement** — on N images: FPGA fc8 == ARM fc8 (must be 100% bit-exact); FPGA top-1 == fp32 torchvision top-1 (quantisation agreement).
4. **Top-1 / Top-5 accuracy** — ImageNet-1k validation set (50,000 images, labels from the official devkit mapped to torchvision class indices) or a fixed, documented subset (e.g. first 2,000 images; state it). Run fp32 torchvision on the host and int8 on the board on the *same* list. Do not calibrate on images you evaluate on.
5. **Resources / timing** — copy the totals from Vivado post-implementation reports.
6. **Power** — Vivado `report_power` is an estimate. Measured board power needs an inline USB/DC power meter: record idle (overlay loaded, no inference) and during a continuous inference loop; report the difference and energy per image = ΔP × latency.

## Result table template (fill in; no values are claimed here)

| Metric | Estimated | Simulated | Synthesised | Measured |
|---|---|---|---|---|
| Clock (MHz) | 100 | — | ☐ (WNS = ☐ ns) | — |
| LUT / FF | 20–30 k / — | ☐ | ☐ / ☐ | — |
| DSP48E1 | ~170 | ☐ | ☐ | — |
| BRAM36 | ~79 (+FIFOs) | ☐ | ☐ | — |
| conv1…conv5 (ms) | 7.1 / 18.2 / 9.8 / 13.0 / 8.7 | ☐ | — | ☐ |
| fc6 / fc7 / fc8 (ms) | 63–94 / 28–42 / 7–10 | ☐ | — | ☐ |
| Accelerator total (ms) | 154–203 | ☐ | — | ☐ |
| CPU pre/post-processing (ms) | — | — | — | ☐ |
| End-to-end latency (ms) | — | — | — | ☐ |
| Throughput (images/s) | 4.9–6.5 (accelerator only) | — | — | ☐ |
| ARM-only int8 NumPy (ms) | — | — | — | ☐ |
| Speed-up vs ARM-only | — | — | — | ☐ |
| FPGA vs ARM fc8 bit-exact | — | — | — | ☐ / N |
| Top-1 / Top-5 fp32 (torchvision, host) | 56.5 / 79.1 (documented) | — | — | ☐ / ☐ on your subset |
| Top-1 / Top-5 int8 FPGA | — | — | — | ☐ / ☐ |
| Board power idle / active (W) | — | — | ☐ (report_power) | ☐ / ☐ |
| Energy per image (J) | — | — | — | ☐ |
