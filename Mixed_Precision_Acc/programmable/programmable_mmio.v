// Command-driven INT8 activation engine. Descriptor layout is documented in README.md.
module programmable_mmio (
    input wire clk, resetn,
    input wire [31:0] command, payload,
    output reg [31:0] reply,
    output wire [31:0] status
);
    localparam IDLE=0, DECODE=1, SETUP=2, LAUNCH=3, DOT=4, SCALE=5,
               ROUND=6, WRITE=7, ADD_READ_A=8, ADD_READ_B=9, ADD_SCALE=10,
               ADD_SUM=11, READBACK=12;

    reg [3:0] state;
    reg ack, busy, done, fault;
    reg [31:0] cycles;
    reg [5:0] pc;
    reg [6:0] count;
    wire request = resetn && command[31] != ack;
    wire [3:0] operation = command[27:24];
    wire [15:0] index = command[15:0];
    wire host_idle = request && !busy && state == IDLE;
    assign status = {28'd0, fault, done, busy, ack};

    (* ram_style = "block" *) reg [7:0] activations [0:65535];
    (* ram_style = "block" *) reg [31:0] weights [0:4095];
    reg signed [31:0] biases [0:255];
    reg signed [63:0] multipliers [0:255];
    reg [31:0] instructions [0:511];
    wire [8:0] descriptor = {pc, 3'b000};
    wire [31:0] flags = instructions[descriptor];
    wire [31:0] addresses = instructions[descriptor + 9'd1];
    wire [31:0] dimensions = instructions[descriptor + 9'd2];
    wire [31:0] output_shape = instructions[descriptor + 9'd3];
    wire [31:0] parameters = instructions[descriptor + 9'd4];
    wire [31:0] add_args = instructions[descriptor + 9'd5];

    reg is_add, w4, relu, stride2, pad;
    reg [15:0] src, dst, skip_src, position, add_last;
    reg [7:0] height, width, channels, out_channels, out_h, out_w;
    reg [7:0] row, col, out_ch, param_index;
    reg [15:0] filter_base, filter_words;
    reg signed [63:0] multiplier, product, scaled_skip;
    reg signed [31:0] biased;
    reg signed [7:0] main_value;
    reg [7:0] output_byte;

    wire act_re, weight_re, dot_done;
    wire [9:0] weight_offset;
    wire signed [31:0] dot_result;
    reg [1:0] ky, kx;
    reg [15:0] plane_base;
    reg padding_d;
    wire [15:0] plane_size = height * width;
    wire signed [9:0] iy = $signed({2'b0, row}) * (stride2 ? 10'sd2 : 10'sd1)
                            + $signed({8'b0, ky}) - $signed({9'b0, pad});
    wire signed [9:0] ix = $signed({2'b0, col}) * (stride2 ? 10'sd2 : 10'sd1)
                            + $signed({8'b0, kx}) - $signed({9'b0, pad});
    wire in_bounds = iy >= 0 && ix >= 0 && iy < $signed({2'b0, height})
                                       && ix < $signed({2'b0, width});
    wire [15:0] conv_address = plane_base + iy[7:0] * width + ix[7:0];
    wire [15:0] read_address = state == ADD_READ_A ? src + position :
                               state == ADD_READ_B ? skip_src + position :
                               state == IDLE ? index : conv_address;
    wire act_read = (act_re && in_bounds) || state == ADD_READ_A ||
                    state == ADD_READ_B || (host_idle && operation == 7);
    wire act_write = state == WRITE || (host_idle && operation == 1);
    wire [15:0] write_address = state == WRITE ? dst + position : index;
    wire [15:0] weight_address = filter_base + {6'b0, weight_offset};
    reg signed [7:0] activation_data;
    reg [31:0] weight_data;
    always @(posedge clk) begin
        if (act_read) activation_data <= activations[read_address];
        if (act_write) activations[write_address] <= state == WRITE ? output_byte : payload[7:0];
        if (weight_re) weight_data <= weights[weight_address[11:0]];
        if (host_idle && operation == 2) weights[index[11:0]] <= payload;
        if (host_idle && operation == 3) biases[index[7:0]] <= payload;
        if (host_idle && operation == 4) multipliers[index[7:0]][31:0] <= payload;
        if (host_idle && operation == 5) multipliers[index[7:0]][63:32] <= payload;
        if (host_idle && operation == 10) instructions[index[8:0]] <= payload;
    end

    wire [15:0] taps = channels * 16'd9;
    wire signed [7:0] dot_activation = padding_d ? 8'sd0 : activation_data;
    dotprod mac (
        .clk(clk), .rst(!resetn), .start(state == LAUNCH), .weight_mode(w4),
        .vector_len(taps), .busy(), .done(dot_done), .result(dot_result),
        .weight_word_reads(), .elapsed_cycles(),
        .act_rd_en(act_re), .act_rd_addr(), .act_rd_data(dot_activation),
        .weight_rd_en(weight_re), .weight_rd_addr(weight_offset), .weight_rd_data(weight_data)
    );

    // Advance input coordinates with the dot-product core's read strobes.
    always @(posedge clk) begin
        if (state == LAUNCH) begin
            ky <= 0;
            kx <= 0;
            plane_base <= src;
        end else if (act_re) begin
            padding_d <= !in_bounds;
            if (kx == 2) begin
                kx <= 0;
                if (ky == 2) begin
                    ky <= 0;
                    plane_base <= plane_base + plane_size;
                end else ky <= ky + 1'b1;
            end else kx <= kx + 1'b1;
        end
    end

    // Add half, subtract one for negative ties, then shift arithmetically.
    wire signed [64:0] extended = {product[63], product};
    wire signed [64:0] rounded = (extended + 65'sd1073741824 -
                                  (product[63] ? 65'sd1 : 65'sd0)) >>> 31;
    wire signed [64:0] minimum = relu ? 65'sd0 : -65'sd128;
    wire last_pixel = col == out_w - 1'b1 && row == out_h - 1'b1;
    wire last_value = is_add ? position == add_last :
                      last_pixel && out_ch == out_channels - 1'b1;

    always @(posedge clk) begin
        if (!resetn) begin
            state <= IDLE;
            ack <= 0;
            busy <= 0;
            done <= 0;
            fault <= 0;
            cycles <= 0;
            reply <= 0;
        end else begin
            if (busy) cycles <= cycles + 1'b1;

            // The host may poll registers while the sequencer runs.
            if (request && state != READBACK && !(host_idle && operation == 7)) begin
                ack <= command[31];
                reply <= operation == 8 ? cycles : 32'd0;
            end

            case (state)
                IDLE: if (host_idle) begin
                    if (operation == 7) state <= READBACK;
                    if (operation == 6) begin
                        done <= 0;
                        fault <= payload == 0 || payload > 64;
                        if (payload != 0 && payload <= 64) begin
                            count <= payload[6:0];
                            pc <= 0;
                            cycles <= 0;
                            busy <= 1;
                            state <= DECODE;
                        end
                    end
                end
                READBACK: begin
                    reply <= {24'd0, activation_data};
                    ack <= command[31];
                    state <= IDLE;
                end
                DECODE: begin
                    is_add <= flags[1:0] == 2;
                    w4 <= flags[2];
                    relu <= flags[3];
                    stride2 <= flags[4];
                    pad <= flags[5];
                    src <= addresses[15:0];
                    dst <= addresses[31:16];
                    width <= dimensions[7:0];
                    height <= dimensions[15:8];
                    channels <= dimensions[23:16];
                    out_channels <= dimensions[31:24];
                    out_w <= output_shape[7:0];
                    out_h <= output_shape[15:8];
                    filter_base <= parameters[15:0];
                    param_index <= parameters[23:16];
                    skip_src <= add_args[15:0];
                    add_last <= add_args[31:16];
                    position <= 0;
                    row <= 0;
                    col <= 0;
                    out_ch <= 0;
                    if (flags[1:0] == 1) state <= SETUP;
                    else if (flags[1:0] == 2) begin
                        multiplier <= {instructions[descriptor + 9'd7],
                                       instructions[descriptor + 9'd6]};
                        state <= ADD_READ_A;
                    end else begin
                        fault <= 1;
                        busy <= 0;
                        state <= IDLE;
                    end
                end
                SETUP: begin
                    multiplier <= multipliers[param_index];
                    filter_words <= w4 ? (taps + 16'd7) >> 3 : (taps + 16'd3) >> 2;
                    state <= LAUNCH;
                end
                LAUNCH: state <= DOT;
                DOT: if (dot_done) begin
                    biased <= dot_result + biases[param_index];
                    state <= SCALE;
                end
                SCALE: begin
                    product <= biased * multiplier;
                    state <= ROUND;
                end
                ADD_READ_A: state <= ADD_READ_B;
                ADD_READ_B: begin
                    main_value <= activation_data;
                    state <= ADD_SCALE;
                end
                ADD_SCALE: begin
                    scaled_skip <= activation_data * multiplier;
                    state <= ADD_SUM;
                end
                ADD_SUM: begin
                    product <= ($signed({{56{main_value[7]}}, main_value}) <<< 31) + scaled_skip;
                    state <= ROUND;
                end
                ROUND: begin
                    output_byte <= rounded < minimum ? minimum[7:0] :
                                   rounded > 65'sd127 ? 8'd127 : rounded[7:0];
                    state <= WRITE;
                end
                WRITE: begin
                    if (last_value) begin
                        if ({1'b0, pc} + 7'd1 == count) begin
                            done <= 1;
                            busy <= 0;
                            state <= IDLE;
                        end else begin
                            pc <= pc + 1'b1;
                            state <= DECODE;
                        end
                    end else begin
                        position <= position + 1'b1;
                        if (is_add) state <= ADD_READ_A;
                        else if (last_pixel) begin
                            row <= 0;
                            col <= 0;
                            out_ch <= out_ch + 1'b1;
                            param_index <= param_index + 1'b1;
                            filter_base <= filter_base + filter_words;
                            state <= SETUP;
                        end else begin
                            if (col == out_w - 1'b1) begin
                                col <= 0;
                                row <= row + 1'b1;
                            end else col <= col + 1'b1;
                            state <= LAUNCH;
                        end
                    end
                end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
