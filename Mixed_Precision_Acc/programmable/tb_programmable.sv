`timescale 1ns / 1ps
module tb_programmable;
    reg clk = 0;
    always #10 clk = !clk;
    reg resetn = 0;
    reg [31:0] command = 0, payload = 0;
    wire [31:0] reply, status;
    reg toggle = 0;
    reg [31:0] result;
    integer i, lane, bit_index, word_index, command_count, program_count, counts_file, scanned;
    reg [31:0] packed_word;
    reg [7:0] image [0:3071];
    reg [31:0] packed_weights [0:53];
    reg [31:0] bias [0:15];
    reg [63:0] mult [0:15];
    reg [7:0] expected [0:16383];
    reg [63:0] requests [0:16095];
    reg [31:0] checks [0:65535];
    string vectors;

    programmable_mmio dut (.clk(clk), .resetn(resetn), .command(command),
                            .payload(payload), .reply(reply), .status(status));

    task issue(input [3:0] op, input [15:0] address, input [31:0] value);
        begin
            @(negedge clk);
            toggle = !toggle;
            payload = value;
            command = {toggle, 3'b000, op, 8'b0, address};
            @(negedge clk);
            while (status[0] !== toggle) @(negedge clk);
            result = reply;
        end
    endtask

    task run_program(input [31:0] instructions);
        begin
            issue(6, 0, instructions);
            while (status[2:1] !== 2'b10 && status[3] !== 1'b1) @(negedge clk);
            if (status[3]) $fatal(1, "Program fault");
        end
    endtask

    task check(input [15:0] address, input [7:0] value);
        begin
            issue(7, address, 0);
            if (result !== {24'b0, value})
                $fatal(1, "Address %0d expected %h, got %h", address, value, result);
        end
    endtask

    initial begin
        repeat (5) @(negedge clk);
        resetn = 1;
        if ($value$plusargs("block=%s", vectors)) begin
            counts_file = $fopen({vectors, "/counts.txt"}, "r");
            if (!counts_file) $fatal(1, "Missing block vector counts");
            scanned = $fscanf(counts_file, "%d %d", command_count, program_count);
            $fclose(counts_file);
            if (scanned != 2 || command_count < 1 || command_count > 16096)
                $fatal(1, "Invalid block vector counts");
            $readmemh({vectors, "/commands.mem"}, requests, 0, command_count - 1);
            $readmemh({vectors, "/expected.mem"}, checks);
            for (i = 0; i < command_count; i = i + 1)
                issue(requests[i][51:48], requests[i][47:32], requests[i][31:0]);
            run_program(program_count);
            for (i = 0; i < 65536; i = i + 1)
                check(checks[i][23:8], checks[i][7:0]);
            issue(8, 0, 0);
            $display("PASS Conv1 and layer1.0 CPU trace: 65536 exact outputs, %0d cycles", result);
        end else begin
            $readmemh("Mixed_precision_firstlayer/input.mem", image);
            $readmemh("Mixed_precision_firstlayer/weights.mem", packed_weights);
            $readmemh("Mixed_precision_firstlayer/bias.mem", bias);
            $readmemh("Mixed_precision_firstlayer/multiplier.mem", mult);
            $readmemh("Mixed_precision_firstlayer/expected.mem", expected);
            for (i = 0; i < 3072; i = i + 1) issue(1, i, image[i]);
            for (i = 0; i < 16; i = i + 1) begin
                for (word_index = 0; word_index < 4; word_index = word_index + 1) begin
                    packed_word = 0;
                    for (lane = 0; lane < 8; lane = lane + 1)
                        if (word_index * 8 + lane < 27) begin
                            bit_index = (i * 27 + word_index * 8 + lane) * 4;
                            packed_word[4*lane +: 4] = packed_weights[bit_index / 32][bit_index % 32 +: 4];
                        end
                    issue(2, i * 4 + word_index, packed_word);
                end
                issue(3, i, bias[i]);
                issue(4, i, mult[i][31:0]);
                issue(5, i, mult[i][63:32]);
            end
            issue(10, 0, 32'h2d);
            issue(10, 1, 32'h40000000);
            issue(10, 2, 32'h10032020);
            issue(10, 3, 32'h2020);
            issue(10, 4, 0);
            for (i = 5; i < 8; i = i + 1) issue(10, i, 0);
            run_program(1);
            for (i = 0; i < 16384; i = i + 1) check(16384 + i, expected[i]);
            issue(8, 0, 0);
            $display("PASS original Conv1 CPU reference: 16384 exact outputs, %0d cycles", result);
        end

        // A second uploaded program uses a 3x5 input, stride 2, no padding,
        // INT8 weights, and signed output. Only the filter center is nonzero.
        for (i = 0; i < 15; i = i + 1) issue(1, i, i - 9);
        issue(1, 96, 4);
        issue(1, 97, 5);
        issue(2, 0, 0);
        issue(2, 1, 32'h00000003);
        issue(2, 2, 0);
        issue(3, 0, 0);
        issue(4, 0, 32'h40000000);
        issue(5, 0, 0);
        issue(10, 0, 32'h11);
        issue(10, 1, 32'h00400000);
        issue(10, 2, 32'h01010305);
        issue(10, 3, 32'h0102);
        issue(10, 4, 0);
        issue(10, 8, 32'h02);
        issue(10, 9, 32'h00800040);
        issue(10, 13, 32'h00010060);
        issue(10, 14, 32'h40000000);
        issue(10, 15, 0);
        run_program(2);
        check(64, 8'hfb); // -4.5 rounds to -5
        check(65, 8'hfe); // -1.5 rounds to -2
        check(128, 8'hfd);
        check(129, 8'h01); // +0.5 rounds to +1
        $display("PASS reprogrammed INT8 stride-2 convolution and scaled add");
        $finish;
    end

    initial begin
        #300000000;
        $fatal(1, "Simulation timeout");
    end
endmodule
