`timescale 1ns / 1ps

module dotprod #(
    parameter ADDR_WIDTH = 10
)(
    input  wire                   clk,
    input  wire                   rst,

    // Control and Configuration
    input  wire                   start,
    input  wire                   weight_mode, // 0 = W8 (INT8), 1 = W4 (INT4)
    input  wire [15:0]            vector_len,  // Runtime vector length (e.g. 64 or 27)

    // Status and Output
    output reg                    busy,
    output reg                    done,
    output reg  signed [31:0]     result,
    output reg  [15:0]            weight_word_reads,
    output reg  [31:0]            elapsed_cycles,

    // Synchronous Activation Memory Interface (1-cycle read latency)
    output reg                    act_rd_en,
    output reg  [ADDR_WIDTH-1:0]  act_rd_addr,
    input  wire signed [7:0]      act_rd_data,

    // Synchronous Weight Memory Interface (1-cycle read latency)
    output reg                    weight_rd_en,
    output reg  [ADDR_WIDTH-1:0]  weight_rd_addr,
    input  wire [31:0]            weight_rd_data
);

    // FSM States
    localparam ST_IDLE    = 2'd0;
    localparam ST_WAIT1   = 2'd1;
    localparam ST_WAIT2   = 2'd2;
    localparam ST_COMPUTE = 2'd3;

    reg [1:0] state;

    // Latched configuration
    reg        mode_reg;
    reg [15:0] len_reg;

    // Internal pipeline registers
    reg [31:0]        weight_reg;
    reg signed [7:0]  act_reg;
    reg [15:0]        term_idx;
    reg [2:0]         sub_idx;
    reg signed [31:0] acc;

    // Internal tracking counters
    reg [15:0] reads_cnt;
    reg [31:0] cycles_cnt;

    // Combinational weight extraction and explicit sign extension
    reg signed [7:0] w_val;
    always @(*) begin
        if (!mode_reg) begin
            // W8 Mode: 4 signed INT8 weights per 32-bit word
            case (sub_idx[1:0])
                2'd0: w_val = weight_reg[7:0];
                2'd1: w_val = weight_reg[15:8];
                2'd2: w_val = weight_reg[23:16];
                2'd3: w_val = weight_reg[31:24];
                default: w_val = 8'sd0;
            endcase
        end else begin
            // W4 Mode: 8 signed INT4 weights per 32-bit word with explicit sign extension
            case (sub_idx[2:0])
                3'd0: w_val = {{4{weight_reg[3]}},  weight_reg[3:0]};
                3'd1: w_val = {{4{weight_reg[7]}},  weight_reg[7:4]};
                3'd2: w_val = {{4{weight_reg[11]}}, weight_reg[11:8]};
                3'd3: w_val = {{4{weight_reg[15]}}, weight_reg[15:12]};
                3'd4: w_val = {{4{weight_reg[19]}}, weight_reg[19:16]};
                3'd5: w_val = {{4{weight_reg[23]}}, weight_reg[23:20]};
                3'd6: w_val = {{4{weight_reg[27]}}, weight_reg[27:24]};
                3'd7: w_val = {{4{weight_reg[31]}}, weight_reg[31:28]};
                default: w_val = 8'sd0;
            endcase
        end
    end

    // Combinational product
    wire signed [15:0] prod;
    assign prod = act_reg * w_val;

    // Prefetch trigger sub-index (2 cycles before word boundary for 1-cycle RAM latency)
    wire [2:0] prefetch_sub_idx;
    assign prefetch_sub_idx = (!mode_reg) ? 3'd1 : 3'd5;

    // Maximum sub-index per word
    wire [2:0] max_sub_idx;
    assign max_sub_idx = (!mode_reg) ? 3'd3 : 3'd7;

    always @(posedge clk) begin
        if (rst) begin
            state             <= ST_IDLE;
            busy              <= 1'b0;
            done              <= 1'b0;
            result            <= 32'sd0;
            weight_word_reads <= 16'd0;
            elapsed_cycles    <= 32'd0;
            act_rd_en         <= 1'b0;
            act_rd_addr       <= {ADDR_WIDTH{1'b0}};
            weight_rd_en      <= 1'b0;
            weight_rd_addr    <= {ADDR_WIDTH{1'b0}};
            term_idx          <= 16'd0;
            sub_idx           <= 3'd0;
            acc               <= 32'sd0;
            reads_cnt         <= 16'd0;
            cycles_cnt        <= 32'd0;
            mode_reg          <= 1'b0;
            len_reg           <= 16'd0;
            weight_reg        <= 32'd0;
            act_reg           <= 8'sd0;
        end else begin
            case (state)
                ST_IDLE: begin
                    act_rd_en    <= 1'b0;
                    weight_rd_en <= 1'b0;

                    // Accept start only if not currently busy
                    if (start && (vector_len == 0 || vector_len > (32'd1 << ADDR_WIDTH))) begin
                        busy              <= 1'b0;
                        done              <= 1'b1;
                        result            <= 32'sd0;
                        weight_word_reads <= 16'd0;
                        elapsed_cycles    <= 32'd1;
                    end else if (start) begin
                        mode_reg          <= weight_mode;
                        len_reg           <= vector_len;
                        busy              <= 1'b1;
                        done              <= 1'b0;
                        cycles_cnt        <= 32'd1;
                        acc               <= 32'sd0;
                        term_idx          <= 16'd0;
                        sub_idx           <= 3'd0;

                        // Request element 0 from both memories
                        weight_rd_en      <= 1'b1;
                        weight_rd_addr    <= {ADDR_WIDTH{1'b0}};
                        reads_cnt         <= 16'd1;

                        act_rd_en         <= 1'b1;
                        act_rd_addr       <= {ADDR_WIDTH{1'b0}};

                        state             <= ST_WAIT1;
                    end
                end

                ST_WAIT1: begin
                    cycles_cnt   <= cycles_cnt + 32'd1;
                    weight_rd_en <= 1'b0;

                    // Request act 1
                    if (len_reg > 16'd1) begin
                        act_rd_en   <= 1'b1;
                        act_rd_addr <= 16'd1;
                    end else begin
                        act_rd_en   <= 1'b0;
                    end

                    state <= ST_WAIT2;
                end

                ST_WAIT2: begin
                    cycles_cnt   <= cycles_cnt + 32'd1;
                    weight_rd_en <= 1'b0;

                    // RAM outputs for word 0 and act 0 are valid now!
                    weight_reg <= weight_rd_data;
                    act_reg    <= act_rd_data;

                    // Request act 2
                    if (len_reg > 16'd2) begin
                        act_rd_en   <= 1'b1;
                        act_rd_addr <= 16'd2;
                    end else begin
                        act_rd_en   <= 1'b0;
                    end

                    state <= ST_COMPUTE;
                end

                ST_COMPUTE: begin
                    cycles_cnt <= cycles_cnt + 32'd1;

                    // Accumulate current product
                    acc <= acc + prod;

                    // Latch activation for next cycle
                    act_reg <= act_rd_data;

                    // Pipeline next activation read
                    if ((term_idx + 17'd3) < len_reg) begin
                        act_rd_en   <= 1'b1;
                        act_rd_addr <= term_idx + 16'd3;
                    end else begin
                        act_rd_en   <= 1'b0;
                    end

                    // Prefetch next weight word 2 cycles before sub_idx wraps
                    if (sub_idx == prefetch_sub_idx) begin
                        if ((term_idx + 17'd3) < len_reg) begin
                            weight_rd_en   <= 1'b1;
                            weight_rd_addr <= weight_rd_addr + 16'd1;
                            reads_cnt      <= reads_cnt + 16'd1;
                        end else begin
                            weight_rd_en   <= 1'b0;
                        end
                    end else begin
                        weight_rd_en <= 1'b0;
                    end

                    // Advance or complete
                    if ((term_idx + 16'd1) >= len_reg) begin
                        result            <= acc + prod; // Include final product
                        done              <= 1'b1;
                        busy              <= 1'b0;
                        weight_word_reads <= reads_cnt;
                        elapsed_cycles    <= cycles_cnt + 32'd1;
                        state             <= ST_IDLE;
                    end else begin
                        term_idx <= term_idx + 16'd1;
                        if (sub_idx == max_sub_idx) begin
                            sub_idx    <= 3'd0;
                            weight_reg <= weight_rd_data;
                        end else begin
                            sub_idx <= sub_idx + 3'd1;
                        end
                    end
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule

