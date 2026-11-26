# create_block_design.tcl -- IP Integrator design "design_1" (sourced by create_project.tcl)
#
#  PS7 M_AXI_GP0 --smc_ctrl--> alexnet_accel_0/s_axi_control   @ 0x4000_0000 (64K)
#                         \--> axi_dma_0/S_AXI_LITE           @ 0x4040_0000 (64K)
#                          \-> axi_intc_0/s_axi               @ 0x4180_0000 (64K)
#  alexnet_accel_0/m_axi_gmem_act --smc_hp0--> PS7 S_AXI_HP0 (DDR, 64-bit)
#  alexnet_accel_0/m_axi_gmem_w   --smc_hp1--> PS7 S_AXI_HP1 (DDR, 64-bit)
#  axi_dma_0 M_AXI_MM2S + M_AXI_S2MM --smc_hp2--> PS7 S_AXI_HP2
#  axi_dma_0/M_AXIS_MM2S --> rtl_stream_0/s_axis ; rtl_stream_0/m_axis --> axi_dma_0/S_AXIS_S2MM
#  interrupts {alexnet, dma mm2s, dma s2mm} --xlconcat--> axi_intc_0 --> PS7 IRQ_F2P
#  ONE clock: FCLK_CLK0 = 100 MHz; ONE reset domain: rst_ps7_0_100M (proc_sys_reset)
#
# Everything is wired explicitly (no connection automation) so no default
# can silently put a 64-bit master on the wrong HP port or an extra clock in.

set bd design_1
create_bd_design $bd

# ---------------------------------------------------------------- PS7 ------
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0]
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
    -config {make_external "FIXED_IO, DDR" apply_board_preset "1" Master "Disable" Slave "Disable"} $ps
set_property -dict [list \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100} \
    CONFIG.PCW_USE_M_AXI_GP0 {1} \
    CONFIG.PCW_USE_S_AXI_HP0 {1} CONFIG.PCW_S_AXI_HP0_DATA_WIDTH {64} \
    CONFIG.PCW_USE_S_AXI_HP1 {1} CONFIG.PCW_S_AXI_HP1_DATA_WIDTH {64} \
    CONFIG.PCW_USE_S_AXI_HP2 {1} CONFIG.PCW_S_AXI_HP2_DATA_WIDTH {64} \
    CONFIG.PCW_USE_FABRIC_INTERRUPT {1} CONFIG.PCW_IRQ_F2P_INTR {1} \
] $ps

# ------------------------------------------------------- clock / reset ----
set clk [get_bd_pins processing_system7_0/FCLK_CLK0]
set rst [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_100M]
connect_bd_net $clk [get_bd_pins rst_ps7_0_100M/slowest_sync_clk]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_100M/ext_reset_in]
set rstn [get_bd_pins rst_ps7_0_100M/peripheral_aresetn]

# ------------------------------------------------------------- IP cells ----
create_bd_cell -type ip -vlnv xilinx.com:hls:alexnet_accel:1.0 alexnet_accel_0

set dma [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0]
set_property -dict [list \
    CONFIG.c_include_sg {0} \
    CONFIG.c_sg_length_width {23} \
    CONFIG.c_include_mm2s {1} CONFIG.c_include_s2mm {1} \
    CONFIG.c_m_axi_mm2s_data_width {32} CONFIG.c_m_axis_mm2s_tdata_width {32} \
    CONFIG.c_m_axi_s2mm_data_width {32} CONFIG.c_s_axis_s2mm_tdata_width {32} \
    CONFIG.c_mm2s_burst_size {16} CONFIG.c_s2mm_burst_size {16} \
] $dma

create_bd_cell -type module -reference rtl_stream_top rtl_stream_0

set intc [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_intc:4.1 axi_intc_0]
set_property -dict [list CONFIG.C_IRQ_CONNECTION {1}] $intc
set cat [create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0]
set_property -dict [list CONFIG.NUM_PORTS {3}] $cat

# ------------------------------------------------------- interconnects ----
proc mk_smc {name nsi nmi} {
    set c [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 $name]
    set_property -dict [list CONFIG.NUM_SI $nsi CONFIG.NUM_MI $nmi] $c
    return $c
}
mk_smc smc_ctrl 1 3
mk_smc smc_hp0  1 1
mk_smc smc_hp1  1 1
mk_smc smc_hp2  2 1

# control path (AXI4-Lite via GP0)
connect_bd_intf_net [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins smc_ctrl/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins smc_ctrl/M00_AXI] [get_bd_intf_pins alexnet_accel_0/s_axi_control]
connect_bd_intf_net [get_bd_intf_pins smc_ctrl/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]
connect_bd_intf_net [get_bd_intf_pins smc_ctrl/M02_AXI] [get_bd_intf_pins axi_intc_0/s_axi]

# data paths (AXI4 masters -> HP ports)
connect_bd_intf_net [get_bd_intf_pins alexnet_accel_0/m_axi_gmem_act] [get_bd_intf_pins smc_hp0/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins smc_hp0/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
connect_bd_intf_net [get_bd_intf_pins alexnet_accel_0/m_axi_gmem_w]   [get_bd_intf_pins smc_hp1/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins smc_hp1/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP1]
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins smc_hp2/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins smc_hp2/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins smc_hp2/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP2]

# AXI4-Stream loop through the handwritten RTL
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S] [get_bd_intf_pins rtl_stream_0/s_axis]
connect_bd_intf_net [get_bd_intf_pins rtl_stream_0/m_axis]   [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]

# ------------------------------------------------------------ clocks -------
foreach p {
    processing_system7_0/M_AXI_GP0_ACLK processing_system7_0/S_AXI_HP0_ACLK
    processing_system7_0/S_AXI_HP1_ACLK processing_system7_0/S_AXI_HP2_ACLK
    alexnet_accel_0/ap_clk
    axi_dma_0/s_axi_lite_aclk axi_dma_0/m_axi_mm2s_aclk axi_dma_0/m_axi_s2mm_aclk
    rtl_stream_0/aclk axi_intc_0/s_axi_aclk
    smc_ctrl/aclk smc_hp0/aclk smc_hp1/aclk smc_hp2/aclk
} { connect_bd_net $clk [get_bd_pins $p] }

# ------------------------------------------------------------ resets -------
foreach p {
    alexnet_accel_0/ap_rst_n axi_dma_0/axi_resetn rtl_stream_0/aresetn
    axi_intc_0/s_axi_aresetn smc_ctrl/aresetn smc_hp0/aresetn smc_hp1/aresetn smc_hp2/aresetn
} { connect_bd_net $rstn [get_bd_pins $p] }

# -------------------------------------------------------- interrupts -------
connect_bd_net [get_bd_pins alexnet_accel_0/interrupt] [get_bd_pins xlconcat_0/In0]
connect_bd_net [get_bd_pins axi_dma_0/mm2s_introut]    [get_bd_pins xlconcat_0/In1]
connect_bd_net [get_bd_pins axi_dma_0/s2mm_introut]    [get_bd_pins xlconcat_0/In2]
connect_bd_net [get_bd_pins xlconcat_0/dout] [get_bd_pins axi_intc_0/intr]
connect_bd_net [get_bd_pins axi_intc_0/irq]  [get_bd_pins processing_system7_0/IRQ_F2P]

# ------------------------------------------------------- address map -------
# Segments are looked up through the interface pin, so no slave-segment name
# (which differs between IPs) is hard-coded.
proc place_slave {intf_pin offset} {
    set seg [get_bd_addr_segs -of_objects [get_bd_intf_pins $intf_pin]]
    assign_bd_address -target_address_space /processing_system7_0/Data $seg \
        -offset $offset -range 64K -force
}
place_slave alexnet_accel_0/s_axi_control 0x40000000
place_slave axi_dma_0/S_AXI_LITE          0x40400000
place_slave axi_intc_0/s_axi              0x41800000
# DDR (512 MB at 0x0) for the HP masters (alexnet gmem_act/gmem_w, DMA MM2S/S2MM)
assign_bd_address
puts "---- address map ----"
foreach sp [get_bd_addr_spaces] {
    foreach s [get_bd_addr_segs -quiet -of_objects $sp] {
        puts [format "%-36s %-60s %s %s" $sp $s [get_property OFFSET $s] [get_property RANGE $s]]
    }
}

regenerate_bd_layout
validate_bd_design
save_bd_design

# ------------------------------------------------------------- wrapper -------
set bd_file [get_files ${bd}.bd]
generate_target all $bd_file
set wrapper [make_wrapper -files $bd_file -top]
add_files -norecurse $wrapper
set_property top ${bd}_wrapper [current_fileset]
update_compile_order -fileset sources_1
puts "INFO: block design created and validated. Next: vivado -mode batch -source vivado/build.tcl"
