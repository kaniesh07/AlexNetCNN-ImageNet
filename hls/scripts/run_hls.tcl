# run_hls.tcl -- Vitis HLS 2024.2 batch flow for the AlexNet layer engine
#
# Run from the hls/ directory:
#   vitis-run --mode hls --tcl scripts/run_hls.tcl            (2024.2 unified CLI)
#   vitis_hls -f scripts/run_hls.tcl                          (legacy command, if installed)
# Choose steps with the environment variable HLS_STEPS (default "csim csynth export"):
#   HLS_STEPS="csim csynth cosim export" vitis-run --mode hls --tcl scripts/run_hls.tcl
# Full-network C simulation additionally needs ALEXNET_GOLDEN=<abs path to build/deploy/golden>.
#
# Outputs
#   alexnet_hls/solution1/syn/report/alexnet_accel_csynth.rpt   (latency, resources)
#   alexnet_hls/solution1/impl/ip/                               (IP repository for Vivado)
#   alexnet_hls/solution1/impl/misc/drivers/alexnet_accel_v1_0/src/xalexnet_accel_hw.h
#                                                                (AUTHORITATIVE register map)

set steps "csim csynth export"
if {[info exists ::env(HLS_STEPS)]} { set steps $::env(HLS_STEPS) }

open_project -reset alexnet_hls
set_top alexnet_accel
set cflags "-I[pwd]/include -std=c++14"
foreach f {src/alexnet_accel.cpp src/conv_engine.cpp src/fc_engine.cpp src/load_store.cpp} {
    add_files $f -cflags $cflags
}
add_files -tb testbench/tb_alexnet_accel.cpp -cflags $cflags

open_solution -reset solution1 -flow_target vivado
set_part xc7z020clg400-1
create_clock -period 10 -name default          ;# 100 MHz = PS FCLK_CLK0
set_clock_uncertainty 12.5%
config_interface -m_axi_addr64=false           ;# Zynq-7000 PS has a 32-bit address map
config_export -format ip_catalog -rtl verilog -vendor xilinx.com -library hls \
              -version 1.0 -display_name "AlexNet Layer Engine"

if {[lsearch $steps csim] >= 0} {
    puts "=== C simulation ==="
    csim_design
}
if {[lsearch $steps csynth] >= 0} {
    puts "=== C synthesis ==="
    csynth_design
}
if {[lsearch $steps cosim] >= 0} {
    puts "=== C/RTL co-simulation (unit-test layers only) ==="
    cosim_design -rtl verilog -trace_level none
}
if {[lsearch $steps export] >= 0} {
    puts "=== Export IP ==="
    export_design -format ip_catalog -rtl verilog
}
exit
