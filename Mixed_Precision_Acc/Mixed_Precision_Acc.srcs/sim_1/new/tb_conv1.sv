`timescale 1ns / 1ps
module tb_conv1;
    reg clk = 0;
    always #5 clk = !clk;
    reg rst = 1, start = 0;
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
    reg [7:0] image [0:3071], expected [0:16383];
    reg [31:0] weights [0:53], biases [0:15];
    reg [63:0] multipliers [0:15];
    string vectors;
    int writes = 0, image_requests = 0, weight_requests = 0;

    conv1 dut (.*);
    always @(posedge clk) begin
        if (image_re) begin
            if (image_addr >= 3072) $fatal(1, "Bad image address %d", image_addr);
            image_data <= image[image_addr];
            image_requests++;
        end
        if (weight_re) begin
            if (weight_addr >= 54) $fatal(1, "Bad weight address %d", weight_addr);
            weight_data <= weights[weight_addr];
            weight_requests++;
        end
        if (param_re) begin
            bias <= biases[param_addr];
            multiplier <= multipliers[param_addr];
        end
        if (out_we) begin
            if (out_addr !== writes[13:0] || writes >= 16384)
                $fatal(1, "Write %d arrived at address %d", writes, out_addr);
            if ($isunknown(expected[writes]) || out_data !== expected[writes])
                $fatal(1, "CHW (%0d,%0d,%0d): expected %0d, got %0d",
                       out_addr[13:10], out_addr[9:5], out_addr[4:0],
                       $signed(expected[writes]), $signed(out_data));
            writes++;
        end
    end
    initial begin
        if (!$value$plusargs("vectors=%s", vectors))
            vectors = "Mixed_precision_firstlayer";
        $readmemh({vectors,"/input.mem"}, image);
        $readmemh({vectors,"/weights.mem"}, weights);
        $readmemh({vectors,"/bias.mem"}, biases);
        $readmemh({vectors,"/multiplier.mem"}, multipliers);
        $readmemh({vectors,"/expected.mem"}, expected);
        repeat (3) @(negedge clk);
        rst = 0;
        start = 1;
        @(negedge clk);
        start = 0;
        wait (done);
        @(negedge clk);
        if (writes != 16384 || output_writes != writes ||
            image_reads != image_requests || image_requests != 424128 ||
            packed_reads != weight_requests || weight_requests != 68)
            $fatal(1, "Traffic mismatch: image=%0d weights=%0d writes=%0d",
                   image_requests, weight_requests, writes);
        $display("PASS %0d outputs, %0d cycles, %0d image reads, %0d packed-word reads",
                 writes, cycles, image_reads, packed_reads);
        $finish;
    end
    initial begin
        #10000000;
        $fatal(1, "Timeout after %d writes", writes);
    end
endmodule
