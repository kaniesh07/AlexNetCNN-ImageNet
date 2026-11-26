// ============================================================================
// mac_unit.v -- signed multiply-accumulate processing element (handwritten)
//
//   en=1, clr=1 : acc <= a*b            (start a new dot product)
//   en=1, clr=0 : acc <= acc + a*b
//   en=0        : acc holds
// Synchronous active-low reset.  With AW=WW=8 the multiply maps to one DSP48E1.
// Overflow: acc wraps modulo 2^ACCW (choose ACCW so it cannot happen: for an
// N-term int8 dot product ACCW >= 16 + ceil(log2(N)) + 1).
// ============================================================================
`timescale 1ns / 1ps
module mac_unit #(
    parameter AW   = 8,
    parameter WW   = 8,
    parameter ACCW = 32
) (
    input  wire                   clk,
    input  wire                   rst_n,
    input  wire                   en,
    input  wire                   clr,
    input  wire signed [AW-1:0]   a,
    input  wire signed [WW-1:0]   b,
    output reg  signed [ACCW-1:0] acc
);
    wire signed [AW+WW-1:0] prod = a * b;
    // explicit sign extension of the product to the accumulator width
    wire signed [ACCW-1:0]  prod_ext = {{(ACCW-AW-WW){prod[AW+WW-1]}}, prod};

    always @(posedge clk) begin
        if (!rst_n)
            acc <= {ACCW{1'b0}};
        else if (en)
            acc <= clr ? prod_ext : acc + prod_ext;
    end
endmodule
