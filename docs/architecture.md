# Architecture, feasibility analysis and HW/SW partition

## 2. AlexNet variant and feasibility on the XC7Z020

### 2.1 Selected model: torchvision AlexNet (`AlexNet_Weights.IMAGENET1K_V1`)

Input **3 × 224 × 224**, ImageNet preprocessing: resize shorter side to 256 (bilinear), centre-crop 224, scale to [0,1], normalise with mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225). Output: 1000 logits, class index *i* = line *i* of `model/imagenet_classes.txt` (generated from the checkpoint's own `meta["categories"]`).

| # | torchvision layer | Kernel / stride / pad | Input → output | Params | MACs |
|---|---|---|---|---|---|
| conv1 | features.0 + ReLU | 64 × 3 × 11 × 11, s4, p2 | 3×224×224 → 64×55×55 | 23,296 | 70.3 M |
| pool1 | features.2 | max 3×3 s2 | → 64×27×27 | | |
| conv2 | features.3 + ReLU | 192 × 64 × 5 × 5, s1, p2 | → 192×27×27 | 307,392 | 224.0 M |
| pool2 | features.5 | max 3×3 s2 | → 192×13×13 | | |
| conv3 | features.6 + ReLU | 384 × 192 × 3 × 3, p1 | → 384×13×13 | 663,936 | 112.1 M |
| conv4 | features.8 + ReLU | 256 × 384 × 3 × 3, p1 | → 256×13×13 | 884,992 | 149.5 M |
| conv5 | features.10 + ReLU | 256 × 256 × 3 × 3, p1 | → 256×13×13 | 590,080 | 99.7 M |
| pool5 | features.12 | max 3×3 s2 | → 256×6×6 | | |
| avgpool | AdaptiveAvgPool2d(6,6) | identity for a 6×6 map | 256×6×6 | | |
| fc6 | classifier.1 + ReLU (dropout = identity in eval) | 9216 → 4096 | | 37,752,832 | 37.7 M |
| fc7 | classifier.4 + ReLU | 4096 → 4096 | | 16,781,312 | 16.8 M |
| fc8 | classifier.6 | 4096 → 1000 | | 4,097,000 | 4.1 M |
| | | | **total** | **61,100,840** | **714 M** |

**Differences from the original Krizhevsky (2012) AlexNet:** that network used 227 × 227 inputs, 96/256/384/384/256 conv channels, two GPU groups (grouped conv2/4/5), and **Local Response Normalisation** after conv1/conv2. torchvision's "one weird trick" variant has 64/192/384/256/256 channels, no groups and **no LRN**, so no LRN engine is needed. The ImageNet class order is torchvision's.

### 2.2 Budget versus device

| Requirement | Size | XC7Z020 / PYNQ-Z2 | Consequence |
|---|---|---|---|
| Weights int8 | 61.1 MB (+0.03 MB conv1 rewrite) | BRAM 0.61 MB | **Weights must stream from DDR** every image |
| Weights fp32 | 244 MB | DDR 512 MB shared with Linux | int8 only on the board |
| Largest activation (int8) | 64×55×55 = 194 KB (conv1 out) | BRAM 0.61 MB | Activations live in DDR, one input map in BRAM per layer |
| Compute | 714 M MAC | 220 DSP48 | ~128 MACs/cycle is the practical ceiling (rest of DSPs: requant, addressing) |
| FC weight traffic | 58.6 MB / image | one HP port 64 bit @100 MHz = 800 MB/s peak | FC layers are **memory-bound**: ≥ 73 ms/image at peak, realistically 100–140 ms |

Fully unrolled per-layer hardware ("one engine per layer") is infeasible: five conv engines + FC engines would need several × the DSPs/BRAM available. **Decision: one time-multiplexed, runtime-configurable layer engine**, called once per layer, operating DDR → BRAM → DDR. All 8 layers (5 conv with their ReLU + 3 max-pools, 3 FC) execute in FPGA fabric; nothing in the network is skipped.

### 2.3 Key implementation idea: conv1 as an exact 3 × 3 convolution
conv1 (11 × 11, stride 4) would need a dedicated strided datapath. Instead the ARM applies a **4 × 4 space-to-depth** to the zero-padded input (3×228×228 → 48×57×57) and the exporter rewrites the weights (zero-padded to 12×12) into a **48-channel 3 × 3 stride-1 conv** producing exactly the same 64×55×55 output (`tests/quantization_test.py` Q1 proves equality in float and integer). Cost: 1.19× MACs on conv1, benefit: the single engine only needs stride 1 with K ∈ {3, 5}.

## 3. Final hardware/software partition

| Stage | Where | Notes |
|---|---|---|
| JPEG decode, resize 256, crop 224, normalise | **ARM (CPU)** | Pillow + NumPy, identical to torchvision |
| Input int8 quantisation, pad 2, space-to-depth, pack | **ARM (CPU)** | ~157 KB, linear-time reshuffle |
| conv1+ReLU+pool1, conv2+ReLU+pool2, conv3+ReLU, conv4+ReLU, conv5+ReLU+pool5 | **FPGA** (layer engine, conv mode) | fused requant/ReLU/max-pool |
| AdaptiveAvgPool(6,6), flatten, dropout | none needed | identity at 224 input; flatten order folded into fc6 weight column order |
| fc6+ReLU, fc7+ReLU, fc8 | **FPGA** (layer engine, FC mode) | fc8 returns raw int32 accumulators |
| fc8 dequantisation (×per-class scale), softmax, top-5 | **ARM (CPU)** | 1000 multiplies |

The CPU-executed work is pre-/post-processing, not network layers. It is timed separately by `pynq/alexnet_inference.py` (expected a few ms for quantise/pack and 10s of ms for JPEG decode/resize on the A9; **measure it**).

## System block diagram

```
                     PS7 (Cortex-A9, PYNQ Linux)                         DDR3 512 MB
        ┌────────────────────────────────────────────────┐            ┌──────────────────┐
        │ M_AXI_GP0 (32b)    S_AXI_HP0  S_AXI_HP1  S_AXI_HP2  IRQ_F2P │◄──►│ weights (8 bufs) │
        └────┬───────────────────▲─────────▲─────────▲────────▲──────┘    │ params  (8 bufs) │
             │ AXI4-Lite         │64b      │64b      │64b     │           │ act ping / pong  │
        ┌────▼─────┐        ┌────┴───┐ ┌───┴────┐ ┌──┴─────┐  │           │ DMA tx / rx bufs │
        │ smc_ctrl │        │smc_hp0 │ │smc_hp1 │ │smc_hp2 │  │           └──────────────────┘
        └┬───┬───┬─┘        └────▲───┘ └───▲────┘ └▲─────▲─┘  │
 0x4000_ │   │   │0x4180_0000    │m_axi    │m_axi   │MM2S │S2MM
 0000    │   │   └──────────┐    │gmem_act │gmem_w  │     │
   ┌─────▼───────────────┐  │  ┌─┴─────────┴──┐  ┌──┴─────┴───┐  M_AXIS_MM2S ┌──────────────┐
   │ alexnet_accel_0     │  │  │ (same IP)    │  │ axi_dma_0  ├─────────────►│ rtl_stream_0 │
   │ Vitis HLS layer     │  │  └──────────────┘  │ simple mode│◄─────────────┤ handwritten  │
   │ engine, s_axi_control│ │                    └──▲─────────┘  S_AXIS_S2MM │ Verilog      │
   └─────────┬───────────┘  │        0x4040_0000 ───┘ S_AXI_LITE              └──────────────┘
             │ interrupt    ▼
             └──► xlconcat ──► axi_intc_0 ──► IRQ_F2P
     clocks: FCLK_CLK0 = 100 MHz to every IP     resets: rst_ps7_0_100M/peripheral_aresetn
```

### Interfaces, clocks, resets, addresses
| Interface | Master → Slave | Width | Address / purpose |
|---|---|---|---|
| `s_axi_control` | PS GP0 → alexnet_accel_0 | AXI4-Lite 32 | 0x4000_0000, 64 KB: ap_ctrl, scalars, buffer addresses, ap_return |
| `m_axi_gmem_act` | alexnet_accel_0 → HP0 | AXI4 64 | DDR 0x0000_0000–0x1FFF_FFFF: read input map, write output map |
| `m_axi_gmem_w` | alexnet_accel_0 → HP1 | AXI4 64 | DDR: read weights + params (FC weight stream) |
| `S_AXI_LITE` | PS GP0 → axi_dma_0 | AXI4-Lite | 0x4040_0000 (PG021 register map) |
| `M_AXI_MM2S`/`M_AXI_S2MM` | axi_dma_0 → HP2 | AXI4 32 | DDR: DMA source/destination buffers |
| `M_AXIS_MM2S` → `s_axis`; `m_axis` → `S_AXIS_S2MM` | DMA ↔ rtl_stream_0 | AXI4-Stream 32 + TLAST | packets for the handwritten RTL |
| `s_axi` | PS GP0 → axi_intc_0 | AXI4-Lite | 0x4180_0000; collects accel + DMA interrupts for PYNQ |

* One clock domain: `FCLK_CLK0` = 100 MHz drives every `aclk/ap_clk` and the PS-side ports `M_AXI_GP0_ACLK`, `S_AXI_HP{0,1,2}_ACLK` (no CDC anywhere).
* One reset domain: `proc_sys_reset` (`rst_ps7_0_100M`) synchronises `FCLK_RESET0_N`; `peripheral_aresetn` drives all IP resets (active low, synchronous).
* Separate HP ports: activations (HP0) and weights (HP1) never contend; the DMA test path (HP2) is isolated from the accelerator.
* Completion: the driver **polls** `ap_done` with a timeout (deterministic, no IRQ plumbing needed). The interrupt is wired through `axi_intc_0` (PYNQ's requirement for Zynq-7000 interrupts) so it can be used with `pynq.Interrupt` later; HLS interrupts stay disabled until IER/GIE are written.

## Layer engine (Vitis HLS) internals

```
DDR act_in ─burst─► in_buf (BRAM, 19,494 × 64b, whole input map, 8 ch/word)
DDR params ─burst─► p_buf  (4096 × 64b: bias | M | shift per output channel)
for each 16-output-channel tile:
   DDR weights ─burst─► w_buf[16][432] (16 BRAM banks)
   MAC loop  II=1 over (oy, ox, cin/8, ky, kx):  16 × dot8 = 128 int8 MAC/clk
             int32 acc (+bias) → requant → ReLU → out_tile (3025 × 128b)
   maxpool   3×3/s2 lane-wise max → pooled (729 × 128b)          [if pool]
   store     2 channel groups → DDR act_out (C8HW)
FC mode:  in_buf ← vector;  stream weights (1 word = 8 MAC / clk) ; requant or raw int32
```
Components (all in `hls/`): `quant_utils.h` (dot8, requant, ReLU, lane max), `load_store.cpp` (weight/bias/activation loading, tiling buffers, store), `conv_engine.cpp` (convolution engine, max-pool engine), `fc_engine.cpp`, `alexnet_accel.cpp` (top, AXI interfaces, validation/status). Complete interface contract: header of `hls/include/alexnet_accel.h`.

### Interface contract (summary)
| Argument | Type | Meaning |
|---|---|---|
| `act_in`, `act_out` | 64-bit word pointers (gmem_act) | input / output activations, layout C8HW |
| `weights`, `params` | 64-bit word pointers (gmem_w) | layer blob, per-channel params |
| `op` | int | 0 conv, 1 FC |
| `cin, hin, win, cout, ksize, pad, hout, wout` | int | conv geometry (stride 1); FC uses `cin`, `cout` |
| `pool, relu, raw_out` | int | fuse 3×3/s2 pool; ReLU; FC raw int32 output |
| return (`ap_return`) | int | 0 OK, 1 bad op, 2 channel multiple, 3 kernel geometry, 4 capacity |

Capacity limits (checked in hardware, `ERR_CAPACITY` otherwise): `(cin/8)·hin·win ≤ 19494`, `(cin/8)·k² ≤ 432`, `hout·wout ≤ 3025`, pooled pixels ≤ 729, `cout ≤ 4096`, `k ≤ 5`, `cin % 8 = 0`, conv `cout % 16 = 0`.

### Register map
Register offsets are **assigned by Vitis HLS** and are not reproduced here as fact. After `export_design`, the authoritative map is
`hls/alexnet_hls/solution1/impl/misc/drivers/alexnet_accel_v1_0/src/xalexnet_accel_hw.h`
(`#define XALEXNET_ACCEL_CONTROL_ADDR_<ARG>_DATA`). `vivado/build.tcl` copies it next to the bitstream; `pynq/hw_header.py` parses it at run time and cross-checks it against the register list in the `.hwh`. Standard `ap_ctrl_hs` bits at `AP_CTRL`: bit0 `ap_start`, bit1 `ap_done` (clear-on-read), bit2 `ap_idle`, bit3 `ap_ready`. `config_interface -m_axi_addr64=false` makes every buffer address a single 32-bit register (the Zynq-7000 address map is 32-bit).

## Handwritten RTL vs HLS-generated RTL
| Module | Origin | Role |
|---|---|---|
| `alexnet_accel` (all its Verilog, AXI4-Lite + 2 × AXI4 master) | **Vitis HLS generated** — `hls/alexnet_hls/solution1/impl/ip/` | the AlexNet compute engine |
| `mac_unit.v`, `relu_unit.v`, `maxpool_unit.v` | handwritten | signed int8 MAC PE, lane ReLU, running max |
| `controller.v` | handwritten | packet FSM sequencing the datapath (conv 3×3 window demo with 4 MAC lanes, requant, ReLU, max-pool) |
| `stream_adapter.v` | handwritten | AXI4-Stream skid buffer: reset + valid/ready handshake |
| `rtl_stream_top.v` | handwritten | AXIS top used as a block-design module reference behind `axi_dma_0` |

The handwritten block has **no AXI4-Lite interface** (in-band packet headers instead), so there is exactly one register map in the design (the HLS one) and no duplicate/conflicting control interfaces. It sits off the inference critical path: it exists to demonstrate and verify handwritten datapath/handshake logic on hardware against the same bit-exact model as the HLS kernel (one test feeds real conv5 activations and expects the HLS pool5 output).

## Resource and performance estimates (category: ESTIMATED — not synthesised)

Produced by `model/perf_model.py` from the loop trip counts. Replace with `csynth` and Vivado reports.

| Resource | Estimate | Device | Basis |
|---|---|---|---|
| DSP48E1 | ~170 (128 MAC + 32 requant + 8 FC + few) | 220 | if HLS maps every 8×8 multiply to a DSP; it may map some to LUTs |
| BRAM36 | ~79 for buffers + m_axi FIFOs/DMA (~10–15) | 140 | 512×72 packing |
| LUT | 20–30 k (adder trees, 16 barrel shifters, muxes, interconnect) | 53.2 k | rough |

| Layer | ms @100 MHz (HP eff. 6 B/clk) | ms (HP eff. 4 B/clk) | Bound |
|---|---|---|---|
| conv1 / conv2 / conv3 / conv4 / conv5 | 7.1 / 18.2 / 9.8 / 13.0 / 8.7 | same | compute |
| fc6 / fc7 / fc8 | 62.9 / 28.0 / 6.8 | 94.4 / 41.9 / 10.2 | DDR |
| **total accelerator** | **≈ 154 ms** | **≈ 203 ms** | |

Bandwidth: 61.5 MB read per image, 99% of it FC weights. Improvement paths if needed after measurement: split FC weights across HP1 + HP2/HP3 (two streams → ~2× FC speed), or keep 4-bit FC weights.

**Risks to watch in synthesis**: (1) II=1 of the MAC loop at 10 ns (accumulate + requant in one stage — HLS will pipeline requant deeper; the `acc` recurrence is a single 32-bit add); (2) DSP count > 220 → apply `#pragma HLS BIND_OP variable=p op=mul impl=fabric` in `dot8` for part of the lanes; (3) BRAM for `in_buf` if HLS rounds depth to a power of two (then 64 BRAM36 instead of 39 — still fits).
