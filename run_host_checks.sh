#!/usr/bin/env bash
# run_host_checks.sh -- every check that runs WITHOUT Vivado/Vitis or the board.
#   ./run_host_checks.sh                 # pretrained weights (downloads once)
#   ./run_host_checks.sh --random-init   # plumbing only, no download
#   CALIB_DIR=calib/ GOLDEN_IMG=cat.jpg ./run_host_checks.sh
set -euo pipefail
cd "$(dirname "$0")"
EXPORT_ARGS="${1:-}"

python3 model/export_weights.py $EXPORT_ARGS
QARGS="--calib-random 8"
[ -n "${CALIB_DIR:-}" ] && QARGS="--calib-dir $CALIB_DIR"
[ -n "${GOLDEN_IMG:-}" ] && QARGS="$QARGS --golden-image $GOLDEN_IMG"
python3 model/quantize_model.py $QARGS
python3 tests/reference_test.py
python3 tests/quantization_test.py
[ -d build/hlsap ] || git clone -q --depth 1 https://github.com/Xilinx/HLS_arbitrary_Precision_Types.git build/hlsap
python3 tests/comparison_test.py
python3 model/perf_model.py
echo "HOST CHECKS COMPLETE"
