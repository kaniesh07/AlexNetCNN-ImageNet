# create_project.tcl -- Vivado 2024.2: project for the PYNQ-Z2 (XC7Z020-1CLG400C)
#
# Prerequisites
#   1. PYNQ-Z2 board files installed (see docs/compatibility.md):
#        <Vivado>/2024.2/data/xhub/boards/XilinxBoardStore/boards/Xilinx/pynq-z2/
#      or set the env var PYNQ_Z2_BOARD_REPO to the folder that contains pynq-z2/
#   2. HLS IP exported:  hls/alexnet_hls/solution1/impl/ip  (hls/scripts/run_hls.tcl)
# Run (from the repository root):
#   vivado -mode batch -source vivado/create_project.tcl
#   vivado -mode batch -source vivado/create_block_design.tcl   (sourced automatically below)

set root      [file normalize [file join [file dirname [info script]] ..]]
set proj_dir  [file join $root build vivado]
set proj_name alexnet_pynq
set hls_ip    [file join $root hls alexnet_hls solution1 impl ip]

if {[info exists ::env(PYNQ_Z2_BOARD_REPO)]} {
    set_param board.repoPaths [list $::env(PYNQ_Z2_BOARD_REPO)]
}

create_project $proj_name $proj_dir -part xc7z020clg400-1 -force

# --- board definition: REQUIRED (it carries the PS7 DDR3/MIO preset) ----------
set bp [lindex [get_board_parts -quiet -latest_file_version {*pynq-z2*}] 0]
if {$bp eq ""} {
    error "PYNQ-Z2 board files not found. Install them (docs/compatibility.md) or set PYNQ_Z2_BOARD_REPO. \
Without them the Zynq PS DDR3 configuration would be wrong and the board would not boot the overlay correctly."
}
puts "INFO: using board part $bp"
set_property board_part $bp [current_project]
set_property target_language Verilog [current_project]

# --- HLS IP repository -----------------------------------------------------
if {![file isdirectory $hls_ip]} {
    error "HLS IP not found at $hls_ip -- run: cd hls && vitis-run --mode hls --tcl scripts/run_hls.tcl"
}
set_property ip_repo_paths [list $hls_ip] [current_project]
update_ip_catalog -rebuild
if {[llength [get_ipdefs -quiet xilinx.com:hls:alexnet_accel:*]] == 0} {
    error "alexnet_accel IP not visible in the catalog after update_ip_catalog"
}

# --- handwritten RTL (synthesisable sources only; testbenches stay out) ------
add_files -norecurse [list \
    [file join $root rtl mac_unit.v] \
    [file join $root rtl relu_unit.v] \
    [file join $root rtl maxpool_unit.v] \
    [file join $root rtl stream_adapter.v] \
    [file join $root rtl controller.v] \
    [file join $root rtl rtl_stream_top.v] ]
update_compile_order -fileset sources_1

source [file join $root vivado create_block_design.tcl]
