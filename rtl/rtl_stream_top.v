// ============================================================================
// rtl_stream_top.v -- handwritten AXI4-Stream post-processing / verification
// block, instantiated in the Vivado block design as a MODULE REFERENCE and
// driven by axi_dma_0 (MM2S -> s_axis, m_axis -> S2MM).
//
//   s_axis --> stream_adapter --> controller (relu/maxpool/mac datapath) -->
//              stream_adapter --> m_axis
//
// This block is NOT on the AlexNet critical path (the HLS layer engine does
// all layer math through DDR); it is the hardware vehicle for the handwritten
// RTL datapath, exercised by pynq/rtl_stream_test.py and compared against the
// same bit-exact reference as the HLS kernel.  It has no AXI4-Lite interface
// on purpose: control is in-band (packet header), so there is no second,
// conflicting register map next to the HLS s_axi_control.
//
// Verilog-2001 only (module references must be Verilog/VHDL).
// ============================================================================
`timescale 1ns / 1ps
module rtl_stream_top (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 aclk CLK" *)
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF s_axis:m_axis, ASSOCIATED_RESET aresetn" *)
    input  wire        aclk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 aresetn RST" *)
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input  wire        aresetn,

    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 s_axis TDATA" *)
    input  wire [31:0] s_axis_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 s_axis TKEEP" *)
    input  wire [3:0]  s_axis_tkeep,      // accepted, unused (DMA always sends full words)
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 s_axis TLAST" *)
    input  wire        s_axis_tlast,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 s_axis TVALID" *)
    input  wire        s_axis_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 s_axis TREADY" *)
    output wire        s_axis_tready,

    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 m_axis TDATA" *)
    output wire [31:0] m_axis_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 m_axis TKEEP" *)
    output wire [3:0]  m_axis_tkeep,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 m_axis TLAST" *)
    output wire        m_axis_tlast,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 m_axis TVALID" *)
    output wire        m_axis_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 m_axis TREADY" *)
    input  wire        m_axis_tready,

    // status (leave unconnected or route to an AXI GPIO / ILA)
    output wire [15:0] packets_done,
    output wire        err_frame
);
    wire [31:0] a_data, c_data;
    wire        a_last, a_valid, a_ready;
    wire        c_last, c_valid, c_ready;
    wire [2:0]  state_dbg;

    assign m_axis_tkeep = 4'hF;

    stream_adapter #(.W(32)) u_in (
        .clk(aclk), .rst_n(aresetn),
        .s_data(s_axis_tdata), .s_last(s_axis_tlast), .s_valid(s_axis_tvalid), .s_ready(s_axis_tready),
        .m_data(a_data), .m_last(a_last), .m_valid(a_valid), .m_ready(a_ready));

    controller u_ctrl (
        .clk(aclk), .rst_n(aresetn),
        .s_data(a_data), .s_last(a_last), .s_valid(a_valid), .s_ready(a_ready),
        .m_data(c_data), .m_last(c_last), .m_valid(c_valid), .m_ready(c_ready),
        .packets_done(packets_done), .err_frame(err_frame), .state_dbg(state_dbg));

    stream_adapter #(.W(32)) u_out (
        .clk(aclk), .rst_n(aresetn),
        .s_data(c_data), .s_last(c_last), .s_valid(c_valid), .s_ready(c_ready),
        .m_data(m_axis_tdata), .m_last(m_axis_tlast), .m_valid(m_axis_tvalid), .m_ready(m_axis_tready));
endmodule
