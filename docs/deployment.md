# Build and deployment guide (dependency order)

Each step has a pass criterion; do not continue until it passes.

## 4. Model export and quantisation (host)
```bash
pip install torch torchvision numpy pillow
python model/export_weights.py                 # downloads IMAGENET1K_V1 (or --checkpoint file.pth)
#   PASS: "numpy float reference matches torchvision AlexNet"; writes build/alexnet_fp32.npz,
#         model/imagenet_classes.txt
python tests/reference_test.py --images my_photos/   # Stage A; eyeball top-5 labels
python model/quantize_model.py --calib-dir calib/ --golden-image my_photos/cat.jpg
#   calib/: 100-500 ImageNet *training* (or val, held out from evaluation) JPEGs
#   writes build/deploy/ (blobs, scales, golden vectors)
EVAL_IMAGES=my_photos python tests/quantization_test.py   # contract tests + int8-vs-fp32 agreement
```
Quantisation choices (all in `model/quantize_model.py`, mirrored bit-exactly in HLS):
* weights int8 symmetric **per output channel**, ±127; activations int8 symmetric **per tensor**, scale from the 99.99th percentile of |activation| over the calibration set (`--percentile 100` = max);
* bias int32 in accumulator units; int32 accumulation (worst-case bound asserted for every layer);
* requant `y = sat8((acc·M + 2^(sh−1)) >> sh)` — round-half-up, saturate to [−128, 127], then ReLU; M is a 15-bit mantissa (relative error < 2⁻¹⁴);
* fc8 is left as int32 and dequantised on the ARM with a per-class float scale.

Trade-off: int8 keeps one DSP per MAC and halves FC traffic vs int16; post-ReLU activations only use the positive half of int8 (7 effective bits). If Top-1 drops more than ~1–2 points vs fp32 on your eval set, the first fix is calibration (more images, percentile 99.9–99.999), the second is unsigned activations (a contract change in HLS + Python).

## 5. HLS (host, Vitis HLS 2024.2)
```bash
cd hls
# optional tool-free pre-check (needs g++ and the open-source ap_int headers):
git clone https://github.com/Xilinx/HLS_arbitrary_Precision_Types.git ../build/hlsap
make -f Makefile.gcc run-full                 # unit tests + full AlexNet vs golden
# Vitis HLS proper:
export ALEXNET_GOLDEN=$PWD/../build/deploy/golden    # enables full-network csim (~minutes)
HLS_STEPS="csim csynth" vitis-run --mode hls --tcl scripts/run_hls.tcl
unset ALEXNET_GOLDEN
HLS_STEPS="cosim export" vitis-run --mode hls --tcl scripts/run_hls.tcl   # cosim = unit-test layers
```
Pass criteria and report reading:
* csim: `ALL TESTS PASSED` and `[full L0..L7] ... PASS`.
* csynth report `alexnet_hls/solution1/syn/report/alexnet_accel_csynth.rpt`:
  - **Timing**: estimated clock ≤ 10 ns minus uncertainty (8.75 ns). If above, see verification.md "Timing".
  - **Loops**: `MAC` (in `conv_tile_mac`) and `FC` (in `fc_layer`) must show *achieved II = 1*. Latency of the top shows "?"/ranges because trip counts are runtime registers — use the per-loop trip × II and the TRIPCOUNT pragmas.
  - **Utilisation**: DSP ≤ ~190, BRAM_18K ≤ ~200 to leave room for DMA/interconnect.
* cosim: `Pass` for all unit cases (full network is too slow for RTL simulation).
* export: `alexnet_hls/solution1/impl/ip/` exists with `component.xml`.

## 6. Handwritten RTL simulation
```bash
cd rtl/testbench
./run_xsim.sh          # Vivado simulator  (./run_iverilog.sh for Icarus Verilog)
# PASS: "ALL RTL TESTS PASSED"
```
All RTL is Verilog-2001 (no SystemVerilog files), synthesisable by Vivado and usable as a block-design module reference.

## 7–8. Vivado block design and bitstream
```bash
source <Xilinx>/Vivado/2024.2/settings64.sh
vivado -mode batch -source vivado/create_project.tcl    # project + block design + wrapper
vivado -mode batch -source vivado/build.tcl             # synth, impl, reports, bit, packaging
# outputs: build/overlay/alexnet.bit  alexnet.hwh  xalexnet_accel_hw.h  (+ alexnet.xsa)
```
What the Tcl does — and how to do it in the GUI if you prefer:
1. **Project** for part `xc7z020clg400-1` with board part `*pynq-z2*`.
2. **Zynq PS** `processing_system7_0`, *Run Block Automation* with *Apply Board Preset* (DDR/MIO/clocks from the board file). Then in *Re-customise IP*: FCLK_CLK0 = 100 MHz; enable M_AXI_GP0; enable S_AXI_HP0/HP1/HP2 (64-bit); enable *Fabric interrupts → IRQ_F2P*.
3. **IP repository**: *Settings → IP → Repository → add* `hls/alexnet_hls/solution1/impl/ip`, then add `alexnet_accel_0`.
4. **AXI DMA** `axi_dma_0`: scatter-gather **off**, buffer-length width 23, MM2S + S2MM, 32-bit memory map and stream widths, burst 16.
5. **RTL**: add the 6 files in `rtl/` (not `testbench/`) as design sources, then right-click canvas → *Add Module* → `rtl_stream_top` (name `rtl_stream_0`).
6. **Interconnect** (SmartConnect): GP0 → {accel `s_axi_control`, DMA `S_AXI_LITE`, `axi_intc_0`}; accel `m_axi_gmem_act` → HP0; accel `m_axi_gmem_w` → HP1; DMA MM2S + S2MM → HP2.
7. **Streams**: DMA `M_AXIS_MM2S` → `rtl_stream_0/s_axis`; `rtl_stream_0/m_axis` → DMA `S_AXIS_S2MM`.
8. **Clocks/resets**: FCLK_CLK0 to every clock pin (including `S_AXI_HPx_ACLK`, `M_AXI_GP0_ACLK`); `proc_sys_reset` driven by FCLK_RESET0_N, its `peripheral_aresetn` to every reset pin.
9. **Interrupts**: accel `interrupt`, DMA `mm2s_introut`, `s2mm_introut` → `xlconcat` (3 ports) → `axi_intc_0/intr`; `axi_intc_0/irq` → PS `IRQ_F2P`.
10. **Address editor**: accel 0x4000_0000, DMA 0x4040_0000, intc 0x4180_0000 (64 KB each); HP masters see DDR 0x0000_0000 (512 MB).
11. **If you use *Run Connection Automation* in the GUI**, check before accepting: each 64-bit HLS master goes to its own HP port (automation may route both to HP0 through one interconnect — functional but halves bandwidth), clock source is FCLK_CLK0 (not "New Clocking Wizard"), and reset is the existing `rst_ps7_0_100M`.
12. **Validate** (F6), **Generate HDL wrapper** (*let Vivado manage*), **Generate Bitstream**.
13. **Timing**: `build/reports/timing_summary.rpt` — WNS ≥ 0 and WHS ≥ 0 (the script stops otherwise). Utilisation in `utilization.rpt`.
14. **Hand-off for PYNQ**: `<proj>.runs/impl_1/design_1_wrapper.bit` and `<proj>.gen/sources_1/bd/design_1/hw_handoff/design_1.hwh`, renamed to the **same base name** (`alexnet.bit` / `alexnet.hwh`).

### Common block-design errors
| Symptom | Cause / fix |
|---|---|
| `[BD 41-758] ... clock pin ... not connected` / validation warns about different clock frequencies | an `S_AXI_HPx_ACLK` or `M_AXI_GP0_ACLK` left unconnected or tied to another clock → connect FCLK_CLK0 |
| `[BD 41-1343] Reset pin ... not associated to clock` on the module reference | `X_INTERFACE_PARAMETER` attributes missing in `rtl_stream_top.v`; re-add the module after editing |
| accelerator not in IP catalog | wrong repo path (must be `.../solution1/impl/ip`), or IP exported for a different part; re-export with `set_part xc7z020clg400-1` |
| `[BD 41-1075]` / unmapped slave / "segment excluded" | run Address Editor → *Assign All*; make sure the HP masters' DDR segments are not excluded |
| DMA validation: data width mismatch | MM2S/S2MM stream width must be 32 to match `rtl_stream_top` |
| Bitstream OK but PYNQ cannot find the IP | the `.hwh` is from another run; always copy `.bit` and `.hwh` from the same build |

## 9. Board deployment (no internet needed on the board)
Copy over SSH/SCP (`xilinx@<board-ip>`, default password `xilinx`) or with the SD card. Target folder e.g. `/home/xilinx/alexnet/`:
```
alexnet/
  alexnet_inference.py accelerator_driver.py hw_header.py preprocessing.py
  alexnet_common.py arm_reference.py rtl_stream_model.py rtl_stream_test.py
  imagenet_classes.txt                         (from model/)
  overlay/alexnet.bit alexnet.hwh xalexnet_accel_hw.h
  deploy/  (whole build/deploy directory, ~62 MB incl. golden/)
  images/  (your test JPEGs)
```
```bash
scp -r pynq/*.py model/imagenet_classes.txt xilinx@pynq:/home/xilinx/alexnet/
scp -r build/overlay build/deploy xilinx@pynq:/home/xilinx/alexnet/
ssh xilinx@pynq
cd /home/xilinx/alexnet
sudo -E python3 rtl_stream_test.py                    # DMA + handwritten RTL
sudo -E python3 alexnet_inference.py --verify-golden  # Stage D, bit-exact per layer
sudo -E python3 alexnet_inference.py --image images/*.jpg --repeat 5 --compare-arm
```
`sudo -E` (or a root Jupyter kernel) is needed because PYNQ maps `/dev/mem` and allocates CMA buffers. Buffers are released by `AlexNetAccel.close()` (context manager) even on exceptions; if a run is killed, `pynq.allocate` memory is returned when the process exits. Dependencies on the board: the PYNQ image's `pynq`, `numpy`, `Pillow` — see `pynq/requirements.txt`.
