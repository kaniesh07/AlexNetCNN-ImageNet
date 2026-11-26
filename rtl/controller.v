// ============================================================================
// controller.v -- packet-driven sequencing FSM + datapath (handwritten)
//
// Processes AXI4-Stream packets of 32-bit words (4 int8 lanes per word).
// Word 0 of every packet is a COMMAND header:
//   [7:0]   op        0 = RELU / pass-through, 1 = MAXPOOL9, 2 = CONV9
//   [8]     relu      apply ReLU to results
//   [20:16] shift     right shift for CONV9 requantisation (0..31)
//   [24]    requant   CONV9: 1 = int8 result (sign-extended), 0 = raw int32 acc
// OP_CONV9 then expects 3 weight words (9 int8 weights, lanes 9..11 ignored)
// and 1 int32 bias word.  The rest of the packet is data; TLAST ends it.
//
//   RELU     : 1 output word per input word, lane-wise ReLU (or identity)
//   MAXPOOL9 : 9 input words = one 3x3 window for 4 channels -> 1 output word
//              (lane-wise signed max, optional ReLU)          == HLS maxpool_tile
//   CONV9    : 3 input words = one 3x3 window (9 int8 pixels, lanes 9..11
//              ignored) -> 1 output word:
//                acc = bias + sum_{i<9} w[i]*x[i]           (4 x mac_unit)
//                requant: y = sat8((acc + 2^(sh-1)) >>> sh) (sh=0: sat8(acc))
//              == HLS requant() with M = 1
// A TLAST that arrives in the middle of a window closes the window early,
// emits its (partial) result with TLAST and sets the sticky err_frame flag,
// so a malformed packet can never hang the AXI DMA S2MM channel.
// ============================================================================
`timescale 1ns / 1ps
module controller (
    input  wire        clk,
    input  wire        rst_n,
    // input stream
    input  wire [31:0] s_data,
    input  wire        s_last,
    input  wire        s_valid,
    output wire        s_ready,
    // output stream
    output reg  [31:0] m_data,
    output reg         m_last,
    output reg         m_valid,
    input  wire        m_ready,
    // status
    output reg  [15:0] packets_done,
    output reg         err_frame,
    output wire [2:0]  state_dbg
);
    localparam OP_RELU = 8'd0, OP_POOL = 8'd1, OP_CONV = 8'd2;

    localparam S_HDR    = 3'd0,
               S_W0     = 3'd1,
               S_W1     = 3'd2,
               S_W2     = 3'd3,
               S_BIAS   = 3'd4,
               S_DATA   = 3'd5,
               S_FINISH = 3'd6;

    reg [2:0]  state;
    reg [7:0]  op;
    reg        relu_en, rq_en;
    reg [4:0]  shift;
    reg [31:0] wreg0, wreg1, wreg2;
    reg signed [31:0] bias;
    reg [3:0]  cnt;          // beat index inside a window
    reg        fin_last;     // window closed by TLAST

    assign state_dbg = state;

    wire out_free = !m_valid || m_ready;
    wire in_data  = (state == S_DATA);
    assign s_ready = (state == S_HDR) || (state == S_W0) || (state == S_W1) ||
                     (state == S_W2) || (state == S_BIAS) || (in_data && out_free);
    wire accept   = s_valid && s_ready;
    wire acc_data = accept && in_data;

    // ---------------- RELU lane unit -----------------------------------------
    wire [31:0] relu_y;
    relu_unit #(.LANES(4)) u_relu (.enable(relu_en), .x(s_data), .y(relu_y));

    // ---------------- MAXPOOL unit -------------------------------------------
    wire [31:0] pool_m, pool_y;
    maxpool_unit #(.LANES(4)) u_pool (
        .clk(clk), .rst_n(rst_n),
        .en(acc_data && op == OP_POOL), .clr(cnt == 4'd0),
        .x(s_data), .m(pool_m));
    relu_unit #(.LANES(4)) u_pool_relu (.enable(relu_en), .x(pool_m), .y(pool_y));

    // ---------------- CONV: 4 MAC lanes ----------------------------------------
    // weight word for the current beat; lanes beyond the 9th weight are masked
    wire [31:0] wsel = (cnt == 4'd0) ? wreg0 : (cnt == 4'd1) ? wreg1 : {24'd0, wreg2[7:0]};
    wire [31:0] xsel = (cnt == 4'd2) ? {24'd0, s_data[7:0]} : s_data;
    wire signed [31:0] macc0, macc1, macc2, macc3;
    wire mac_en = acc_data && (op == OP_CONV);
    mac_unit #(.AW(8), .WW(8), .ACCW(32)) u_mac0 (.clk(clk), .rst_n(rst_n), .en(mac_en), .clr(cnt == 4'd0),
        .a(xsel[7:0]),   .b(wsel[7:0]),   .acc(macc0));
    mac_unit #(.AW(8), .WW(8), .ACCW(32)) u_mac1 (.clk(clk), .rst_n(rst_n), .en(mac_en), .clr(cnt == 4'd0),
        .a(xsel[15:8]),  .b(wsel[15:8]),  .acc(macc1));
    mac_unit #(.AW(8), .WW(8), .ACCW(32)) u_mac2 (.clk(clk), .rst_n(rst_n), .en(mac_en), .clr(cnt == 4'd0),
        .a(xsel[23:16]), .b(wsel[23:16]), .acc(macc2));
    mac_unit #(.AW(8), .WW(8), .ACCW(32)) u_mac3 (.clk(clk), .rst_n(rst_n), .en(mac_en), .clr(cnt == 4'd0),
        .a(xsel[31:24]), .b(wsel[31:24]), .acc(macc3));

    // final sum + requantisation (combinational, registered into m_data)
    wire signed [33:0] conv_sum = macc0 + macc1 + macc2 + macc3 + bias;
    wire signed [33:0] rnd      = (shift == 5'd0) ? 34'sd0 : (34'sd1 <<< (shift - 5'd1));
    wire signed [33:0] shifted  = (conv_sum + rnd) >>> shift;
    wire signed [7:0]  sat8     = (shifted > 34'sd127)  ? 8'sd127 :
                                  (shifted < -34'sd128) ? -8'sd128 : shifted[7:0];
    wire signed [7:0]  conv_q   = (relu_en && sat8[7]) ? 8'sd0 : sat8;
    wire signed [31:0] conv_raw = conv_sum[31:0];
    wire [31:0] conv_y = rq_en ? {{24{conv_q[7]}}, conv_q}
                               : ((relu_en && conv_raw[31]) ? 32'd0 : conv_raw);

    wire [3:0] win_len = (op == OP_POOL) ? 4'd9 : 4'd3;

    // ---------------- FSM ---------------------------------------------------------
    always @(posedge clk) begin
        if (!rst_n) begin
            state        <= S_HDR;
            op           <= OP_RELU;
            relu_en      <= 1'b0;
            rq_en        <= 1'b0;
            shift        <= 5'd0;
            wreg0        <= 32'd0;
            wreg1        <= 32'd0;
            wreg2        <= 32'd0;
            bias         <= 32'sd0;
            cnt          <= 4'd0;
            fin_last     <= 1'b0;
            m_data       <= 32'd0;
            m_last       <= 1'b0;
            m_valid      <= 1'b0;
            packets_done <= 16'd0;
            err_frame    <= 1'b0;
        end else begin
            // default: retire the output beat once it has been accepted
            if (m_valid && m_ready) m_valid <= 1'b0;

            case (state)
            S_HDR: if (accept) begin
                op      <= s_data[7:0];
                relu_en <= s_data[8];
                shift   <= s_data[20:16];
                rq_en   <= s_data[24];
                cnt     <= 4'd0;
                if (s_last) begin                       // header-only packet
                    err_frame    <= 1'b1;
                    packets_done <= packets_done + 16'd1;
                end else if (s_data[7:0] == OP_CONV)
                    state <= S_W0;
                else if (s_data[7:0] == OP_RELU || s_data[7:0] == OP_POOL)
                    state <= S_DATA;
                else begin
                    err_frame <= 1'b1;                  // unknown op: pass-through
                    op        <= OP_RELU;
                    relu_en   <= 1'b0;
                    state     <= S_DATA;
                end
            end
            S_W0:   if (accept) begin wreg0 <= s_data; state <= s_last ? S_HDR : S_W1; end
            S_W1:   if (accept) begin wreg1 <= s_data; state <= s_last ? S_HDR : S_W2; end
            S_W2:   if (accept) begin wreg2 <= s_data; state <= s_last ? S_HDR : S_BIAS; end
            S_BIAS: if (accept) begin bias  <= s_data; state <= s_last ? S_HDR : S_DATA; end
            S_DATA: if (accept) begin
                if (op == OP_RELU) begin
                    m_data  <= relu_y;
                    m_last  <= s_last;
                    m_valid <= 1'b1;
                    if (s_last) begin
                        state        <= S_HDR;
                        packets_done <= packets_done + 16'd1;
                    end
                end else begin
                    if (s_last && cnt != win_len - 4'd1) err_frame <= 1'b1;
                    if (s_last || cnt == win_len - 4'd1) begin
                        fin_last <= s_last;
                        cnt      <= 4'd0;
                        state    <= S_FINISH;
                    end else begin
                        cnt <= cnt + 4'd1;
                    end
                end
            end
            S_FINISH: if (out_free) begin
                // the unit registers now hold the complete window result
                m_data  <= (op == OP_POOL) ? pool_y : conv_y;
                m_last  <= fin_last;
                m_valid <= 1'b1;
                if (fin_last) begin
                    state        <= S_HDR;
                    packets_done <= packets_done + 16'd1;
                end else
                    state <= S_DATA;
            end
            default: state <= S_HDR;
            endcase
        end
    end
endmodule
