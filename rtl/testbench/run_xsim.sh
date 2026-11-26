#!/usr/bin/env bash
# run_xsim.sh -- Vivado 2024.2 simulator (xvlog/xelab/xsim) for the handwritten RTL.
# Source Vivado first:  source <Vivado>/2024.2/settings64.sh
# Usage: cd rtl/testbench && ./run_xsim.sh
set -euo pipefail
cd "$(dirname "$0")"
RTL=..
python3 gen_vectors.py

# --- leaf units
xvlog --nolog $RTL/mac_unit.v $RTL/relu_unit.v $RTL/maxpool_unit.v tb_units.v
xelab --nolog -debug off tb_units -s tb_units_sim
xsim --nolog tb_units_sim -R | tee units.log

# --- streaming top (vec_params.vh / *.hex are read from this directory)
xvlog --nolog -i . $RTL/mac_unit.v $RTL/relu_unit.v $RTL/maxpool_unit.v $RTL/stream_adapter.v \
      $RTL/controller.v $RTL/rtl_stream_top.v tb_rtl_stream_top.v
xelab --nolog -debug typical tb_rtl_stream_top -s tb_top_sim
xsim --nolog tb_top_sim -R | tee top.log     # add --gui instead of -R for waveforms

grep -q "TB_RESULT: PASS" units.log && grep -q "TB_RESULT: PASS" top.log && echo "ALL RTL TESTS PASSED"
