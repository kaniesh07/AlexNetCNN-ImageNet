# AlexNet on the PYNQ-Z2 — Vitis HLS layer engine + handwritten Verilog + PYNQ

Complete ImageNet AlexNet inference (torchvision `AlexNet_Weights.IMAGENET1K_V1`, 224×224 → 1000 class scores) on the PYNQ-Z2 (Zynq-7000 XC7Z020-1CLG400C), built with Vivado / Vitis HLS 2024.2.

**All 5 conv layers (with ReLU and the 3 max-pools) and all 3 FC layers run in the FPGA fabric** on a single time-multiplexed int8 layer engine (128 MAC/cycle) that streams weights and activations through DDR. The ARM does only image pre-processing (resize/crop/normalise, input quantisation, an exact space-to-depth rewrite for conv1) and post-processing (fc8 dequantisation, softmax, top-5). The board-side flow is plain PYNQ Python; no Vitis AI / DPU is used.

```
image ─► ARM: resize/crop/normalise ─► int8 + space-to-depth ─► DDR
          FPGA: conv1 ─ conv2 ─ conv3 ─ conv4 ─ conv5 ─ fc6 ─ fc7 ─ fc8   (8 invocations of alexnet_accel)
          ARM: int32 logits × scale ─► softmax ─► top-5 ImageNet labels
```

## Status — read this first
| | |
|---|---|
| Verified here (no AMD tools) | numpy reference = torchvision; preprocessing = torchvision; all quantisation-contract tests; **HLS C++ kernel compiled with g++ runs the full 8-layer network bit-exactly against the Python golden model**; handwritten Verilog passes self-checking testbenches (incl. a real-activation packet checked against the HLS output); PYNQ driver passes against a behavioural mock of the IP |
| **Not yet verified** | Vitis HLS csynth/cosim, Vivado implementation/timing, bitstream, anything on the board, accuracy with the real pretrained weights (sandbox could not reach the checkpoint server; all runs used torchvision's architecture with random-init weights) |
| Performance | **estimates only**: ≈ 154–203 ms accelerator time per image at 100 MHz, FC layers DDR-bandwidth-bound (`model/perf_model.py`) |

## Where each required section lives
| # | Section | Location |
|---|---|---|
| 1 | Compatibility & installation checklist | `docs/compatibility.md` |
| 2 | AlexNet architecture & resource feasibility | `docs/architecture.md` §2 |
| 3 | HW/SW partition, block diagram, interfaces | `docs/architecture.md` §3 |
| 4 | Quantisation & weight export | `model/export_weights.py`, `model/quantize_model.py`, `docs/deployment.md` §4 |
| 5 | HLS C++ + testbench | `hls/` , `docs/deployment.md` §5 |
| 6 | Handwritten Verilog + testbenches | `rtl/`, `rtl/testbench/` |
| 7 | Vivado block design | `vivado/create_block_design.tcl`, `docs/deployment.md` §7 |
| 8 | Tcl build scripts | `vivado/*.tcl`, `hls/scripts/run_hls.tcl` |
| 9 | PYNQ deployment | `pynq/`, `docs/deployment.md` §9 |
| 10 | Verification & debugging | `docs/verification.md`, `tests/` |
| 11 | Performance methodology + result template | `docs/performance.md` |
| 12 | Reproducibility checklist | below |

## Project structure
```
alexnet_pynq/
├── README.md                 run_host_checks.sh (all tool-free checks)
├── docs/   compatibility.md architecture.md deployment.md verification.md performance.md
├── model/  export_weights.py quantize_model.py perf_model.py imagenet_classes.txt
├── hls/
│   ├── include/ alexnet_accel.h (interface contract) quant_utils.h engines.h
│   ├── src/     alexnet_accel.cpp conv_engine.cpp fc_engine.cpp load_store.cpp
│   ├── testbench/ tb_alexnet_accel.cpp
│   ├── scripts/ run_hls.tcl          Makefile.gcc (g++ csim without Vitis)
├── rtl/    mac_unit.v relu_unit.v maxpool_unit.v stream_adapter.v controller.v rtl_stream_top.v
│   └── testbench/ tb_units.v tb_rtl_stream_top.v gen_vectors.py run_xsim.sh run_iverilog.sh
├── vivado/ create_project.tcl create_block_design.tcl build.tcl
├── pynq/   alexnet_inference.py accelerator_driver.py hw_header.py preprocessing.py
│           alexnet_common.py (numerical contract) arm_reference.py
│           rtl_stream_model.py rtl_stream_test.py requirements.txt
└── tests/  reference_test.py quantization_test.py comparison_test.py driver_mock_test.py mock_pynq/
Generated (not shipped): build/alexnet_fp32.npz build/deploy/ build/overlay/ build/vivado/ hls/alexnet_hls/
Generated-IP files: hls/alexnet_hls/solution1/impl/ip/ (Vivado IP repo),
                    .../impl/misc/drivers/alexnet_accel_v1_0/src/xalexnet_accel_hw.h (register map)
```

## Quick start (dependency order)
```bash
./run_host_checks.sh                                   # Stage A/B/C on the host (torch needed)
cd hls && vitis-run --mode hls --tcl scripts/run_hls.tcl && cd ..        # csim, csynth, export IP
(cd rtl/testbench && ./run_xsim.sh)                    # RTL in Vivado simulator
vivado -mode batch -source vivado/create_project.tcl   # needs PYNQ-Z2 board files
vivado -mode batch -source vivado/build.tcl            # -> build/overlay/alexnet.{bit,hwh}
# copy to the board (docs/deployment.md §9), then on the board:
sudo -E python3 rtl_stream_test.py
sudo -E python3 alexnet_inference.py --verify-golden
sudo -E python3 alexnet_inference.py --image images/cat.jpg --compare-arm
```

## 12. Reproducibility checklist
- [ ] Host: Vivado + Vitis HLS **2024.2**, PYNQ-Z2 board files installed, `get_board_parts *pynq-z2*` non-empty
- [ ] Board: PYNQ image version recorded (`/etc/pynq_version`), `CmaFree` ≥ 80 MB after boot
- [ ] Weights: `alexnet-owt-7be5be79.pth` (IMAGENET1K_V1); record torch/torchvision versions printed by `export_weights.py`
- [ ] Calibration image list and `--percentile` recorded (`build/deploy/model_meta.json` stores scales and percentile)
- [ ] `tests/quantization_test.py` and `tests/comparison_test.py` all PASS
- [ ] HLS csim (incl. full network) PASS; csynth: `MAC` and `FC` loops II = 1; estimated clock ≤ 8.75 ns
- [ ] cosim PASS
- [ ] RTL `run_xsim.sh` → ALL RTL TESTS PASSED
- [ ] Vivado: `validate_bd_design` clean; WNS ≥ 0, WHS ≥ 0; utilisation recorded
- [ ] `alexnet.bit`, `alexnet.hwh`, `xalexnet_accel_hw.h` from the **same** build copied together
- [ ] Board: `rtl_stream_test.py` PASS; `--verify-golden` bit-exact for all 8 layers
- [ ] `--compare-arm` FPGA fc8 bit-exact with ARM
- [ ] Results table in `docs/performance.md` filled with categories kept separate
- [ ] Git commit hash / archive checksum of this project recorded with the results

## Key design decisions
* **One reusable engine, 8 invocations** — the only way to fit full AlexNet in 220 DSP / 140 BRAM.
* **conv1 via space-to-depth** — exact, lets one stride-1 engine serve all conv layers.
* **C8HW int8 layout everywhere** — 64-bit words map 1:1 onto the HP ports and the 8-wide dot product; the fc6 weight columns are permuted at export so flatten costs nothing.
* **Bit-exact integer contract** in one Python file shared by host tools, board code, HLS testbench and RTL vectors.
* **Register map never hard-coded** — parsed from the HLS-generated header and cross-checked against the `.hwh`.
* **Handwritten RTL** for the MAC PE, ReLU/max-pool datapath, AXIS handshake/skid logic and sequencing FSM, verified in simulation and on hardware through AXI DMA, without a second AXI4-Lite register map.
