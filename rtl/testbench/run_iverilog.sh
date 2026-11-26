#!/usr/bin/env bash
# run_iverilog.sh -- open-source simulation of the handwritten RTL (Icarus Verilog >= 11)
# Usage: cd rtl/testbench && ./run_iverilog.sh
set -euo pipefail
cd "$(dirname "$0")"
RTL=..
python3 gen_vectors.py
iverilog -g2001 -Wall -o tb_units.vvp tb_units.v $RTL/mac_unit.v $RTL/relu_unit.v $RTL/maxpool_unit.v
vvp -n tb_units.vvp | tee units.log
iverilog -g2001 -Wall -o tb_top.vvp tb_rtl_stream_top.v $RTL/rtl_stream_top.v $RTL/stream_adapter.v \
    $RTL/controller.v $RTL/mac_unit.v $RTL/relu_unit.v $RTL/maxpool_unit.v
vvp -n tb_top.vvp | tee top.log
grep -q "TB_RESULT: PASS" units.log && grep -q "TB_RESULT: PASS" top.log && echo "ALL RTL TESTS PASSED"
