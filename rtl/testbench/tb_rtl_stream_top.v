// ============================================================================
// tb_rtl_stream_top.v -- self-checking testbench for rtl_stream_top (Verilog-2001)
//
//  * 100 MHz clock, synchronous active-low reset held for 10 cycles
//  * input driver: replays in_vec.hex with random TVALID gaps
//  * output sink : random TREADY back-pressure, compares every beat (data and
//                  TLAST) with exp_vec.hex (golden model = pynq/rtl_stream_model.py)
//  * protocol assertions on BOTH stream interfaces:
//      - TVALID/TDATA/TLAST must stay stable while TVALID && !TREADY
//      - no TVALID during reset
//  * watchdog timeout
// Run:  see rtl/testbench/run_iverilog.sh and run_xsim.sh
// ============================================================================
`timescale 1ns / 1ps
`include "vec_params.vh"

module tb_rtl_stream_top;
    reg clk = 1'b0;
    reg rstn = 1'b0;
    always #5 clk = ~clk;            // 100 MHz

    reg  [31:0] s_tdata;
    reg         s_tlast, s_tvalid;
    wire        s_tready;
    wire [31:0] m_tdata;
    wire [3:0]  m_tkeep;
    wire        m_tlast, m_tvalid;
    reg         m_tready;
    wire [15:0] packets_done;
    wire        err_frame;

    rtl_stream_top dut (
        .aclk(clk), .aresetn(rstn),
        .s_axis_tdata(s_tdata), .s_axis_tkeep(4'hF), .s_axis_tlast(s_tlast),
        .s_axis_tvalid(s_tvalid), .s_axis_tready(s_tready),
        .m_axis_tdata(m_tdata), .m_axis_tkeep(m_tkeep), .m_axis_tlast(m_tlast),
        .m_axis_tvalid(m_tvalid), .m_axis_tready(m_tready),
        .packets_done(packets_done), .err_frame(err_frame));

    reg [32:0] in_mem  [0:`N_IN-1];
    reg [32:0] exp_mem [0:`N_OUT-1];
    integer in_idx, out_idx, errors, cycles, seed;

    initial begin
        $readmemh("in_vec.hex", in_mem);
        $readmemh("exp_vec.hex", exp_mem);
    end

    // ---------------- input driver (random valid gaps) --------------------
    always @(posedge clk) begin
        if (!rstn) begin
            s_tvalid <= 1'b0;
            s_tdata  <= 32'd0;
            s_tlast  <= 1'b0;
            in_idx   <= 0;
        end else begin
            if (s_tvalid && s_tready) in_idx <= in_idx + 1;
            // keep the beat while it has not been accepted (AXIS rule)
            if (!s_tvalid || s_tready) begin
                if ((s_tvalid && s_tready ? in_idx + 1 : in_idx) < `N_IN && ($random(seed) & 3) != 0) begin
                    s_tvalid <= 1'b1;
                    {s_tlast, s_tdata} <= in_mem[s_tvalid && s_tready ? in_idx + 1 : in_idx];
                end else begin
                    s_tvalid <= 1'b0;
                end
            end
        end
    end

    // ---------------- output sink (random back-pressure) + checker -----
    always @(posedge clk) begin
        if (!rstn) begin
            m_tready <= 1'b0;
            out_idx  <= 0;
        end else begin
            m_tready <= (($random(seed) % 3) != 0);
            if (m_tvalid && m_tready) begin
                if (out_idx >= `N_OUT) begin
                    errors = errors + 1;
                    $display("ERROR: unexpected extra output beat %h", m_tdata);
                end else if ({m_tlast, m_tdata} !== exp_mem[out_idx]) begin
                    errors = errors + 1;
                    if (errors < 10)
                        $display("ERROR beat %0d: got last=%b data=%h  exp last=%b data=%h",
                                 out_idx, m_tlast, m_tdata, exp_mem[out_idx][32], exp_mem[out_idx][31:0]);
                end
                out_idx <= out_idx + 1;
            end
        end
    end

    // ---------------- protocol assertions -----------------------------------
    reg        m_hold, s_hold;
    reg [32:0] m_prev, s_prev;
    always @(posedge clk) begin
        if (!rstn) begin
            m_hold <= 1'b0; s_hold <= 1'b0;
            if (m_tvalid) begin errors = errors + 1; $display("ASSERT: m_axis_tvalid high in reset"); end
        end else begin
            if (m_hold && (!m_tvalid || {m_tlast, m_tdata} !== m_prev)) begin
                errors = errors + 1;
                $display("ASSERT @%0t: m_axis changed/dropped while stalled", $time);
            end
            if (s_hold && (!s_tvalid || {s_tlast, s_tdata} !== s_prev)) begin
                errors = errors + 1;
                $display("ASSERT @%0t: s_axis driver violated AXIS hold rule", $time);
            end
            m_hold <= m_tvalid && !m_tready;  m_prev <= {m_tlast, m_tdata};
            s_hold <= s_tvalid && !s_tready;  s_prev <= {s_tlast, s_tdata};
            if (m_tkeep !== 4'hF) begin errors = errors + 1; $display("ASSERT: tkeep != F"); end
        end
    end

    // ---------------- sequence / watchdog ----------------------------------
    initial begin
        errors = 0; cycles = 0; seed = 32'h1234;
        repeat (10) @(posedge clk);
        rstn = 1'b1;
        while ((out_idx < `N_OUT || in_idx < `N_IN) && cycles < 200000) begin
            @(posedge clk);
            cycles = cycles + 1;
        end
        repeat (20) @(posedge clk);
        if (cycles >= 200000) begin
            errors = errors + 1;
            $display("ERROR: timeout (in %0d/%0d, out %0d/%0d)", in_idx, `N_IN, out_idx, `N_OUT);
        end
        if (packets_done != `N_PKT) begin
            errors = errors + 1;
            $display("ERROR: packets_done=%0d expected %0d", packets_done, `N_PKT);
        end
        if (err_frame) begin
            errors = errors + 1;
            $display("ERROR: err_frame set on well-formed packets");
        end
        $display("in beats %0d, out beats %0d / %0d, packets %0d, cycles %0d",
                 in_idx, out_idx, `N_OUT, packets_done, cycles);
        if (errors == 0) $display("TB_RESULT: PASS");
        else             $display("TB_RESULT: FAIL (%0d errors)", errors);
        $finish;
    end
endmodule
