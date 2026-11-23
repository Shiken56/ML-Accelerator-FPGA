module conv1 (
    input wire clk, rst, start,
    output reg busy, done,
    output wire image_re,
    output wire [11:0] image_addr,
    input wire signed [7:0] image_data,
    output wire weight_re,
    output wire [5:0] weight_addr,
    input wire [31:0] weight_data,
    output wire param_re,
    output wire [3:0] param_addr,
    input wire signed [31:0] bias,
    input wire signed [63:0] multiplier,
    output wire out_we,
    output reg [13:0] out_addr,
    output wire [7:0] out_data,
    output reg [31:0] cycles, image_reads, packed_reads, output_writes
);
    localparam IDLE=0, FETCH=1, CAPTURE=2, LAUNCH=3, DOT=4, SCALE=5, WRITE=6;
    reg [2:0] state, word_index;
    reg [159:0] filter_words;
    wire [8:0] filter_start = param_addr * 9'd27;
    wire [2:0] nibble = filter_start[2:0];
    wire [2:0] last_word = nibble > 5 ? 3'd4 : 3'd3;
    assign param_addr = out_addr[13:10];
    assign weight_addr = filter_start[8:3] + word_index;
    assign weight_re = state == FETCH;
    assign param_re = weight_re && word_index == 0;

    wire act_re, local_weight_re, dot_done;
    wire [4:0] tap, local_weight_addr;
    wire signed [31:0] dot_result;
    reg [31:0] local_weight_data;
    reg is_padding;
    wire [1:0] input_channel = tap / 5'd9;
    wire [3:0] spatial_tap = tap % 5'd9;
    wire [5:0] row = {1'b0, out_addr[9:5]} + spatial_tap / 4'd3 - 6'd1;
    wire [5:0] col = {1'b0, out_addr[4:0]} + spatial_tap % 4'd3 - 6'd1;
    assign image_addr = {input_channel, row[4:0], col[4:0]};
    assign image_re = act_re && !row[5] && !col[5];
    wire signed [7:0] activation = is_padding ? 8'sd0 : image_data;
    wire [4:0] aligned_nibble = {local_weight_addr[1:0], 3'b0} + nibble;

    // The padding flag follows the same latency as the image RAM.
    always @(posedge clk) begin
        if (act_re) is_padding <= !image_re;
        if (local_weight_re) local_weight_data <= filter_words[4*aligned_nibble +: 32];
    end

    dotprod #(.ADDR_WIDTH(5)) mac (
        .clk(clk), .rst(rst), .start(state == LAUNCH), .weight_mode(1'b1),
        .vector_len(16'd27), .busy(), .done(dot_done), .result(dot_result),
        .weight_word_reads(), .elapsed_cycles(),
        .act_rd_en(act_re), .act_rd_addr(tap), .act_rd_data(activation),
        .weight_rd_en(local_weight_re), .weight_rd_addr(local_weight_addr),
        .weight_rd_data(local_weight_data)
    );

    // Negative rounded results map to zero after ReLU.
    reg signed [31:0] biased;
    reg signed [63:0] product;
    wire [63:0] rounded = ($unsigned(product) + 64'd1073741824) >> 31;
    assign out_data = product <= 0 ? 8'd0 : (rounded > 127 ? 8'd127 : rounded[7:0]);
    assign out_we = state == WRITE;

    always @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            busy <= 0;
            done <= 0;
            out_addr <= 0;
            word_index <= 0;
            cycles <= 0;
            image_reads <= 0;
            packed_reads <= 0;
            output_writes <= 0;
        end else begin
            if (busy) cycles <= cycles + 1;
            if (image_re) image_reads <= image_reads + 1;
            if (weight_re) packed_reads <= packed_reads + 1;
            if (out_we) output_writes <= output_writes + 1;
            case (state)
                IDLE: if (start) begin
                    busy <= 1;
                    done <= 0;
                    out_addr <= 0;
                    word_index <= 0;
                    filter_words <= 0;
                    cycles <= 0;
                    image_reads <= 0;
                    packed_reads <= 0;
                    output_writes <= 0;
                    state <= FETCH;
                end
                FETCH: state <= CAPTURE;
                CAPTURE: begin
                    filter_words[32*word_index +: 32] <= weight_data;
                    if (word_index == last_word) state <= LAUNCH;
                    else begin
                        word_index <= word_index + 1;
                        state <= FETCH;
                    end
                end
                LAUNCH: state <= DOT;
                DOT: if (dot_done) begin
                    biased <= dot_result + bias;
                    state <= SCALE;
                end
                SCALE: begin
                    product <= biased * multiplier;
                    state <= WRITE;
                end
                WRITE: begin
                    if (&out_addr) begin
                        busy <= 0;
                        done <= 1;
                        state <= IDLE;
                    end else begin
                        out_addr <= out_addr + 1'b1;
                        if (&out_addr[9:0]) begin
                            word_index <= 0;
                            filter_words <= 0;
                            state <= FETCH;
                        end else state <= LAUNCH;
                    end
                end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
