// Command-driven INT8 activation engine. Descriptor layout is documented in README.md.
module programmable_mmio (
    input wire clk, resetn,
    input wire [31:0] command, payload,
    output reg [31:0] reply,
    output wire [31:0] status
);
    localparam IDLE=0, DECODE=1, SETUP=2, LAUNCH=3, DOT=4, SCALE=5,
               ROUND=6, WRITE=7, ADD_READ_A=8, ADD_READ_B=9, ADD_SCALE=10,
               ADD_SUM=11, READBACK=12, PROJECT_READ=13, PROJECT_CAPTURE=14,
               POOL_READ=15, POOL_ACCUMULATE=16, POOL_ROUND=17, WRITE_RAW=18;

    reg [4:0] state;
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
    (* ram_style = "block" *) reg [31:0] weights [0:65535];
    (* ram_style = "block" *) reg signed [31:0] biases [0:1023];
    (* ram_style = "block" *) reg [31:0] multipliers_low [0:1023];
    (* ram_style = "block" *) reg [31:0] multipliers_high [0:1023];
    reg signed [31:0] classifier_result [0:255];
    reg [31:0] instructions [0:511];
    wire [8:0] descriptor = {pc, 3'b000};
    wire [31:0] flags = instructions[descriptor];
    wire [31:0] addresses = instructions[descriptor + 9'd1];
    wire [31:0] dimensions = instructions[descriptor + 9'd2];
    wire [31:0] output_shape = instructions[descriptor + 9'd3];
    wire [31:0] parameters = instructions[descriptor + 9'd4];
    wire [31:0] add_args = instructions[descriptor + 9'd5];

    reg is_add, is_pool, is_project, is_linear, binary, w4, relu, stride2, pad;
    reg [15:0] src, dst, skip_src, position, add_last;
    reg [7:0] height, width, channels, out_channels, out_h, out_w;
    reg [7:0] row, col, out_ch, channel_padding;
    reg [9:0] param_index;
    reg [15:0] filter_base, filter_words;
    reg signed [63:0] multiplier, product, scaled_skip;
    reg signed [31:0] filter_bias, conv_multiplier_low, conv_multiplier_high;
    reg signed [31:0] biased;
    reg signed [7:0] main_value;
    reg [7:0] output_byte;

    wire act_re, weight_re, dot_done;
    wire [9:0] weight_offset;
    wire signed [31:0] dot_result;
    reg [1:0] ky, kx;
    reg [15:0] plane_base;
    reg padding_d;
    reg [15:0] project_plane, pool_address, pool_pixel, pool_last;
    reg [3:0] pool_shift;
    reg signed [31:0] pool_sum;
    wire [15:0] plane_size = height * width;
    wire signed [9:0] iy = $signed({2'b0, row}) * (stride2 ? 10'sd2 : 10'sd1)
                            + $signed({8'b0, ky}) - $signed({9'b0, pad});
    wire signed [9:0] ix = $signed({2'b0, col}) * (stride2 ? 10'sd2 : 10'sd1)
                            + $signed({8'b0, kx}) - $signed({9'b0, pad});
    wire in_bounds = is_linear || (iy >= 0 && ix >= 0 && iy < $signed({2'b0, height})
                                       && ix < $signed({2'b0, width}));
    wire [15:0] conv_address = is_linear ? plane_base :
                               plane_base + iy[7:0] * width + ix[7:0];
    wire project_valid = out_ch >= channel_padding &&
                         {1'b0, out_ch} < {1'b0, channel_padding} + {1'b0, channels};
    wire [15:0] project_address = project_plane + ({row, 1'b0} * width) + {col, 1'b0};
    wire [15:0] read_address = state == ADD_READ_A ? src + position :
                               state == ADD_READ_B ? skip_src + position :
                               state == PROJECT_READ ? project_address :
                               state == POOL_READ ? pool_address :
                               state == IDLE ? index : conv_address;
    wire act_read = (act_re && in_bounds) || state == ADD_READ_A ||
                    state == ADD_READ_B || state == POOL_READ ||
                    (state == PROJECT_READ && project_valid) || (host_idle && operation == 7);
    wire act_write = state == WRITE || (host_idle && operation == 1);
    wire [15:0] write_address = state == WRITE ? dst + position : index;
    wire [15:0] weight_address = filter_base +
                                 (binary ? {8'b0, weight_offset[9:2]} : {6'b0, weight_offset});
    reg signed [7:0] activation_data;
    reg [31:0] weight_data;
    reg [1:0] binary_group;
    wire [7:0] binary_bits = weight_data[8*binary_group +: 8];
    wire [31:0] expanded_weights;
    genvar lane;
    generate for (lane = 0; lane < 8; lane = lane + 1) begin : expand_binary
        assign expanded_weights[4*lane +: 4] = binary_bits[lane] ? 4'hf : 4'h1;
    end endgenerate

    always @(posedge clk) begin
        if (act_read) activation_data <= activations[read_address];
        if (act_write) activations[write_address] <= state == WRITE ? output_byte : payload[7:0];
        if (weight_re) begin
            weight_data <= weights[weight_address];
            binary_group <= weight_offset[1:0];
        end
        if (state == SETUP) begin
            filter_bias <= biases[param_index];
            conv_multiplier_low <= multipliers_low[param_index];
            conv_multiplier_high <= multipliers_high[param_index];
        end
        if (state == WRITE_RAW) classifier_result[out_ch] <= biased;
        if (host_idle && operation == 2) weights[index] <= payload;
        if (host_idle && operation == 3) biases[index[9:0]] <= payload;
        if (host_idle && operation == 4) multipliers_low[index[9:0]] <= payload;
        if (host_idle && operation == 5) multipliers_high[index[9:0]] <= payload;
        if (host_idle && operation == 10) instructions[index[8:0]] <= payload;
    end

    wire [15:0] taps = is_linear ? {8'b0, channels} : channels * 16'd9;
    wire signed [7:0] dot_activation = padding_d ? 8'sd0 : activation_data;
    dotprod mac (
        .clk(clk), .rst(!resetn), .start(state == LAUNCH), .weight_mode(w4 || binary),
        .vector_len(taps), .busy(), .done(dot_done), .result(dot_result),
        .weight_word_reads(), .elapsed_cycles(),
        .act_rd_en(act_re), .act_rd_addr(), .act_rd_data(dot_activation),
        .weight_rd_en(weight_re), .weight_rd_addr(weight_offset),
        .weight_rd_data(binary ? expanded_weights : weight_data)
    );

    // Advance input coordinates with the dot-product core's read strobes.
    always @(posedge clk) begin
        if (state == LAUNCH) begin
            ky <= 0;
            kx <= 0;
            plane_base <= src;
        end else if (act_re) begin
            padding_d <= !in_bounds;
            if (is_linear) plane_base <= plane_base + 1'b1;
            else if (kx == 2) begin
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
    wire signed [31:0] averaged = (pool_sum + $signed((32'd1 << pool_shift) >> 1)) >>> pool_shift;
    wire last_pixel = col == out_w - 1'b1 && row == out_h - 1'b1;
    wire last_value = is_add ? position == add_last :
                      is_pool ? out_ch == out_channels - 1'b1 :
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
                if (host_idle && operation == 11) reply <= classifier_result[index[7:0]];
                if (host_idle && operation == 12) reply <= 32'h4d504132;
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
                    is_pool <= flags[1:0] == 3 && !flags[2];
                    is_project <= flags[1:0] == 3 && flags[2];
                    is_linear <= flags[7];
                    binary <= flags[6];
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
                    param_index <= parameters[25:16];
                    channel_padding <= parameters[7:0];
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
                    end else if (flags[1:0] == 3) begin
                        if (flags[2]) begin
                            project_plane <= addresses[15:0];
                            state <= PROJECT_READ;
                        end else begin
                            pool_address <= addresses[15:0];
                            pool_pixel <= 0;
                            pool_last <= dimensions[15:8] * dimensions[7:0] - 16'd1;
                            pool_shift <= output_shape[19:16];
                            pool_sum <= 0;
                            state <= POOL_READ;
                        end
                    end else begin
                        fault <= 1;
                        busy <= 0;
                        state <= IDLE;
                    end
                end
                SETUP: begin
                    filter_words <= binary ? (taps + 16'd31) >> 5 :
                                    w4 ? (taps + 16'd7) >> 3 : (taps + 16'd3) >> 2;
                    state <= LAUNCH;
                end
                LAUNCH: state <= DOT;
                DOT: if (dot_done) begin
                    biased <= dot_result + filter_bias;
                    state <= is_linear ? WRITE_RAW : SCALE;
                end
                SCALE: begin
                    product <= biased * $signed({conv_multiplier_high, conv_multiplier_low});
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
                PROJECT_READ: state <= PROJECT_CAPTURE;
                PROJECT_CAPTURE: begin
                    output_byte <= project_valid ? activation_data : 8'd0;
                    state <= WRITE;
                end
                POOL_READ: state <= POOL_ACCUMULATE;
                POOL_ACCUMULATE: begin
                    pool_sum <= pool_sum + activation_data;
                    pool_address <= pool_address + 1'b1;
                    pool_pixel <= pool_pixel + 1'b1;
                    state <= pool_pixel == pool_last ? POOL_ROUND : POOL_READ;
                end
                POOL_ROUND: begin
                    output_byte <= averaged[7:0];
                    state <= WRITE;
                end
                ROUND: begin
                    output_byte <= rounded < minimum ? minimum[7:0] :
                                   rounded > 65'sd127 ? 8'd127 : rounded[7:0];
                    state <= WRITE;
                end
                WRITE, WRITE_RAW: begin
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
                        else if (is_pool) begin
                            out_ch <= out_ch + 1'b1;
                            pool_sum <= 0;
                            pool_pixel <= 0;
                            state <= POOL_READ;
                        end
                        else if (is_project) begin
                            if (last_pixel) begin
                                row <= 0;
                                col <= 0;
                                out_ch <= out_ch + 1'b1;
                                if (project_valid) project_plane <= project_plane + plane_size;
                            end else if (col == out_w - 1'b1) begin
                                col <= 0;
                                row <= row + 1'b1;
                            end else col <= col + 1'b1;
                            state <= PROJECT_READ;
                        end
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
