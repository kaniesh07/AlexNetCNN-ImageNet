// ============================================================================
// maxpool_unit.v -- lane-parallel running signed max (handwritten)
//
// Feed the WINDOW elements of a pooling window one beat at a time (each beat
// carries the same window position for LANES channels):
//   en=1, clr=1 : m <= x            (first element of a window)
//   en=1, clr=0 : m <= max(m, x)    (lane-wise, signed int8)
// For AlexNet's 3x3/stride-2 pooling the window is 9 beats; the controller
// counts beats.  Equivalent to maxpool_tile() in hls/src/conv_engine.cpp.
// ============================================================================
`timescale 1ns / 1ps
module maxpool_unit #(
    parameter LANES = 4
) (
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 en,
    input  wire                 clr,
    input  wire [8*LANES-1:0]   x,
    output reg  [8*LANES-1:0]   m
);
    integer l;
    reg [8*LANES-1:0] nxt;
    always @* begin
        for (l = 0; l < LANES; l = l + 1)
            nxt[8*l +: 8] = ($signed(x[8*l +: 8]) > $signed(m[8*l +: 8])) ? x[8*l +: 8] : m[8*l +: 8];
    end

    always @(posedge clk) begin
        if (!rst_n)
            m <= {8*LANES{1'b0}};
        else if (en)
            m <= clr ? x : nxt;
    end
endmodule
