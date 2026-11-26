// ============================================================================
// tb_units.v -- self-checking unit tests for the handwritten leaf modules
//   mac_unit     : 2000 random int8 MAC sequences incl. -128*-128, vs behavioural model
//   relu_unit    : all 256 int8 values, enable on/off
//   maxpool_unit : 500 random windows of 9 beats x 4 lanes
// Verilog-2001.
// ============================================================================
`timescale 1ns / 1ps
module tb_units;
    reg clk = 0, rstn = 0;
    always #5 clk = ~clk;
    integer errors = 0, i, j, seed = 7;

    // ---------------- mac_unit ----------------
    reg en, clr;
    reg signed [7:0] a, b;
    wire signed [31:0] acc;
    reg signed [31:0] model;
    mac_unit #(.AW(8), .WW(8), .ACCW(32)) u_mac (.clk(clk), .rst_n(rstn), .en(en), .clr(clr), .a(a), .b(b), .acc(acc));

    // ---------------- relu_unit ---------------
    reg  [31:0] rx;
    reg         ren;
    wire [31:0] ry;
    relu_unit #(.LANES(4)) u_relu (.enable(ren), .x(rx), .y(ry));

    // ---------------- maxpool_unit ------------
    reg        pen, pclr;
    reg [31:0] px;
    wire [31:0] pm;
    reg signed [7:0] pmodel [0:3];
    maxpool_unit #(.LANES(4)) u_pool (.clk(clk), .rst_n(rstn), .en(pen), .clr(pclr), .x(px), .m(pm));

    integer l, k, n;
    reg signed [7:0] v;
    initial begin
        en = 0; clr = 0; a = 0; b = 0; pen = 0; pclr = 0; px = 0; ren = 0; rx = 0;
        repeat (3) @(posedge clk);
        if (acc !== 0 || pm !== 0) begin errors = errors + 1; $display("reset value wrong"); end
        rstn = 1;

        // ---- MAC ----
        for (i = 0; i < 2000; i = i + 1) begin
            n = 1 + ($random(seed) & 15);
            for (j = 0; j < n; j = j + 1) begin
                @(negedge clk);
                en = 1; clr = (j == 0);
                a = (i == 0) ? -8'sd128 : $random(seed);
                b = (i == 0) ? -8'sd128 : $random(seed);
                model = (j == 0) ? a * b : model + a * b;
                if (($random(seed) & 7) == 0) begin   // random stall: en=0 must hold
                    @(negedge clk); en = 0;
                end
            end
            @(negedge clk); en = 0;
            if (acc !== model) begin
                errors = errors + 1;
                if (errors < 5) $display("MAC mismatch seq %0d: got %0d exp %0d", i, acc, model);
            end
        end
        $display("mac_unit     : %s", errors ? "FAIL" : "PASS");

        // ---- ReLU: exhaustive per lane ----
        k = errors;
        for (i = 0; i < 256; i = i + 1) begin
            for (j = 0; j < 2; j = j + 1) begin
                ren = j; rx = {i[7:0], ~i[7:0], i[7:0] ^ 8'h80, 8'h00};
                #1;
                for (l = 0; l < 4; l = l + 1) begin
                    v = rx[8*l +: 8];
                    if (ry[8*l +: 8] !== ((ren && v < 0) ? 8'd0 : rx[8*l +: 8])) errors = errors + 1;
                end
            end
        end
        $display("relu_unit    : %s", (errors - k) ? "FAIL" : "PASS");

        // ---- maxpool ----
        k = errors;
        for (i = 0; i < 500; i = i + 1) begin
            for (j = 0; j < 9; j = j + 1) begin
                @(negedge clk);
                pen = 1; pclr = (j == 0); px = $random(seed);
                for (l = 0; l < 4; l = l + 1) begin
                    v = px[8*l +: 8];
                    if (j == 0 || v > pmodel[l]) pmodel[l] = v;
                end
            end
            @(negedge clk); pen = 0;
            for (l = 0; l < 4; l = l + 1)
                if ($signed(pm[8*l +: 8]) !== pmodel[l]) begin
                    errors = errors + 1;
                    if (errors - k < 5) $display("POOL mismatch win %0d lane %0d: %0d vs %0d", i, l, $signed(pm[8*l +: 8]), pmodel[l]);
                end
        end
        $display("maxpool_unit : %s", (errors - k) ? "FAIL" : "PASS");

        if (errors == 0) $display("TB_RESULT: PASS");
        else             $display("TB_RESULT: FAIL (%0d errors)", errors);
        $finish;
    end
endmodule
