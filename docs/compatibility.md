# 1. Compatibility and installation checklist

Two separate environments are involved. Nothing from the host toolchain is installed on the board, and the board never needs internet access.

| | **Host PC (build)** | **PYNQ-Z2 (runtime)** |
|---|---|---|
| OS | Linux (Ubuntu 22.04/24.04 or RHEL as supported by AMD 2024.2) or Windows 10/11 | PYNQ Linux SD image (Ubuntu-based, ARM Cortex-A9) |
| FPGA tools | Vivado 2024.2, Vitis HLS 2024.2 (installed by the Vitis/Vivado 2024.2 unified installer) | none |
| Python | ≥3.9 with `torch`, `torchvision`, `numpy`, `Pillow` (model export/quantisation/tests) | PYNQ's Python 3 with `pynq`, `numpy`, `Pillow` (all preinstalled on the image) |
| Optional | `g++` + [AMD ap_int headers](https://github.com/Xilinx/HLS_arbitrary_Precision_Types) and `iverilog` for tool-free C/RTL checks | — |
| Produces / consumes | `alexnet.bit`, `alexnet.hwh`, `xalexnet_accel_hw.h`, `deploy/*.npy` | consumes those files |

## Checklist (verify each item, do not assume)

- [ ] **Device.** PYNQ-Z2 = XC7Z020-1CLG400C. Resources used for planning: 53,200 LUTs, 106,400 FFs, 140 × 36 Kb BRAM (4.9 Mb), 220 DSP48E1 (AMD Zynq-7000 product tables). Vivado part string: `xc7z020clg400-1`.
- [ ] **Vivado/Vitis HLS 2024.2 installed** with Zynq-7000 device support selected in the installer.
- [ ] **HLS command line.** In 2024.2 the HLS tool is part of the Vitis unified installation. Run `vitis-run -h` and confirm `--mode hls --tcl` is listed; the scripts here are plain HLS Tcl and run with either `vitis-run --mode hls --tcl scripts/run_hls.tcl` or, if your install still provides it, `vitis_hls -f scripts/run_hls.tcl`.
- [ ] **PYNQ-Z2 board files.** Download from TUL (link on the PYNQ "Board Settings" documentation page), extract and copy the `pynq-z2` folder to `<Xilinx>/Vivado/2024.2/data/xhub/boards/XilinxBoardStore/boards/Xilinx/` (or point `PYNQ_Z2_BOARD_REPO` at the folder that contains it). Restart Vivado. Check in the Tcl console: `get_board_parts *pynq-z2*` returns a part. `vivado/create_project.tcl` refuses to continue without it because the board preset supplies the PS7 DDR3/MIO configuration.
- [ ] **PYNQ image on the board.** Check `cat /etc/pynq_version` (or `pip3 show pynq`). PYNQ v3.1 publishes a PYNQ-Z2 image and its own overlays were built with **Vivado 2024.1**; v3.0.1 used Vivado 2022.1. A 2024.2 bitstream is not tied to the image's tool version (the PL is programmed from the `.bit` through the Linux FPGA manager, and PYNQ parses the `.hwh`), but **this pairing has not been tested here** — Stage D (`--verify-golden`) is the acceptance test. If `.hwh` parsing fails, see `docs/verification.md` → "Overlay load".
- [ ] **CMA (contiguous memory) on the board.** The design needs ≈ 62 MB of physically contiguous buffers (largest single block: fc6 weights 37.7 MB). Check `grep -i cma /proc/meminfo`. If `CmaFree` is too small, increase the `cma=` kernel argument in the boot configuration of your image, or run inference right after boot.
- [ ] **Python packages on the host:** `pip install torch torchvision numpy pillow` (tested here with torch 2.14 / torchvision 0.29 / numpy 2.5; any version providing `AlexNet_Weights.IMAGENET1K_V1` works).
- [ ] **Weights.** `torchvision.models.AlexNet_Weights.IMAGENET1K_V1` (`alexnet-owt-7be5be79.pth`, 61,100,840 parameters, documented acc@1 56.522 / acc@5 79.066). The export script downloads it, or loads a local copy with `--checkpoint` for offline machines.

## Things that are NOT used
- **Vitis AI / DPU**: not used and not assumed to exist on the PYNQ-Z2. Vitis **HLS** (C++ → RTL) is a different tool.
- No PYNQ prebuilt overlay is required (the base overlay is not modified).
