// AXI GPIO channels expose a command/data pair and a reply/status pair.
// A changed command MSB submits exactly one operation.
module conv1_mmio (
    input wire clk, resetn,
    input wire [31:0] command, payload,
    output reg [31:0] reply,
    output wire [31:0] status
);
    wire rst = !resetn;
    wire busy, done, image_re, weight_re, param_re, out_we;
    wire [11:0] image_addr;
    wire [5:0] weight_addr;
    wire [3:0] param_addr;
    wire [13:0] out_addr;
    wire [7:0] out_data;
    wire [31:0] cycles, image_reads, packed_reads, output_writes;
    reg signed [7:0] image_data;
    reg [31:0] weight_data;
    reg signed [31:0] bias;
    reg signed [63:0] multiplier;
    reg ack, output_pending;
    reg [7:0] output_byte;
    wire request = !rst && command[31] != ack && !output_pending;
    wire [3:0] operation = command[27:24];
    wire [13:0] index = command[13:0];
    wire start = request && operation == 6 && !busy;
    assign status = {29'd0, done, busy, ack};

    (* ram_style = "block" *) reg [7:0] image [0:3071];
    (* ram_style = "block" *) reg [7:0] output_ram [0:16383];
    reg [31:0] weights [0:53], biases [0:15];
    reg [31:0] multiplier_low [0:15], multiplier_high [0:15];

    conv1 core (
        .clk(clk), .rst(rst), .start(start), .busy(busy), .done(done),
        .image_re(image_re), .image_addr(image_addr), .image_data(image_data),
        .weight_re(weight_re), .weight_addr(weight_addr), .weight_data(weight_data),
        .param_re(param_re), .param_addr(param_addr), .bias(bias), .multiplier(multiplier),
        .out_we(out_we), .out_addr(out_addr), .out_data(out_data),
        .cycles(cycles), .image_reads(image_reads), .packed_reads(packed_reads),
        .output_writes(output_writes)
    );

    always @(posedge clk) begin
        if (image_re) image_data <= image[image_addr];
        if (weight_re) weight_data <= weights[weight_addr];
        if (param_re) begin
            bias <= biases[param_addr];
            multiplier <= {multiplier_high[param_addr], multiplier_low[param_addr]};
        end
        if (out_we) output_ram[out_addr] <= out_data;
        if (request && operation == 7 && !busy) output_byte <= output_ram[index];
        if (request && !busy) begin
            case (operation)
                1: if (index < 3072) image[index[11:0]] <= payload[7:0];
                2: if (index < 54) weights[index[5:0]] <= payload;
                3: if (index < 16) biases[index[3:0]] <= payload;
                4: if (index < 16) multiplier_low[index[3:0]] <= payload;
                5: if (index < 16) multiplier_high[index[3:0]] <= payload;
                default: ;
            endcase
        end
        if (rst) begin
            ack <= 0;
            output_pending <= 0;
            reply <= 0;
        end else if (output_pending) begin
            output_pending <= 0;
            reply <= {24'd0, output_byte};
            ack <= command[31];
        end else if (request) begin
            if (operation == 7 && !busy)
                output_pending <= 1;
            else begin
                ack <= command[31];
                case (operation)
                    8: reply <= cycles;
                    9: reply <= image_reads;
                    10: reply <= packed_reads;
                    11: reply <= output_writes;
                    default: reply <= 0;
                endcase
            end
        end
    end
endmodule
