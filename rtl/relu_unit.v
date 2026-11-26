// ============================================================================
// relu_unit.v -- lane-parallel int8 ReLU (purely combinational, handwritten)
//   y[l] = (enable && x[l] < 0) ? 0 : x[l]       for each of LANES int8 lanes
// ============================================================================
`timescale 1ns / 1ps
module relu_unit #(
    parameter LANES = 4
) (
    input  wire                 enable,
    input  wire [8*LANES-1:0]   x,
    output wire [8*LANES-1:0]   y
);
    genvar l;
    generate
        for (l = 0; l < LANES; l = l + 1) begin : g_lane
            wire neg = x[8*l+7];
            assign y[8*l +: 8] = (enable && neg) ? 8'sd0 : x[8*l +: 8];
        end
    endgenerate
endmodule
