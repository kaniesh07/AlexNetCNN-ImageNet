// ============================================================================
// stream_adapter.v -- AXI4-Stream register slice / skid buffer (handwritten)
//
// Full-throughput (1 beat/clock) valid/ready pipeline stage that registers
// BOTH directions: m_valid/m_data/m_last come from flops and s_ready comes
// from a flop, so no combinational path crosses the adapter.  Used at the
// input and output of rtl_stream_top to isolate the AXI DMA from the
// processing FSM.
//
// AXI4-Stream rules honoured:
//   * a beat transfers when valid && ready on the same rising edge
//   * once m_valid is high, m_data/m_last stay stable until m_ready
//   * m_valid is low during and after reset (synchronous active-low rst_n)
// ============================================================================
`timescale 1ns / 1ps
module stream_adapter #(
    parameter W = 32
) (
    input  wire         clk,
    input  wire         rst_n,
    // upstream (slave) side
    input  wire [W-1:0] s_data,
    input  wire         s_last,
    input  wire         s_valid,
    output wire         s_ready,
    // downstream (master) side
    output wire [W-1:0] m_data,
    output wire         m_last,
    output wire         m_valid,
    input  wire         m_ready
);
    reg [W:0] main_d, skid_d;      // {last, data}
    reg       main_v, skid_v;

    assign s_ready          = ~skid_v;
    assign m_valid          = main_v;
    assign {m_last, m_data} = main_d;

    always @(posedge clk) begin
        if (!rst_n) begin
            main_v <= 1'b0;
            skid_v <= 1'b0;
            main_d <= {(W+1){1'b0}};
            skid_d <= {(W+1){1'b0}};
        end else begin
            if (m_ready || !main_v) begin
                // output register is free (or being emptied this cycle)
                if (skid_v) begin
                    main_d <= skid_d;
                    main_v <= 1'b1;
                    skid_v <= 1'b0;
                end else begin
                    main_v <= s_valid;
                    if (s_valid) main_d <= {s_last, s_data};
                end
            end else if (s_valid && !skid_v) begin
                // output stalled: park the accepted beat in the skid register
                skid_d <= {s_last, s_data};
                skid_v <= 1'b1;
            end
        end
    end
endmodule
