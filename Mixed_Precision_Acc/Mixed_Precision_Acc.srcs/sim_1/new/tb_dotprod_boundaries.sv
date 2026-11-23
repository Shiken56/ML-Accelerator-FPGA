`timescale 1ns / 1ps
module tb_dotprod_boundaries;
    reg clk = 0;
    always #5 clk = !clk;
    reg rst = 1, start = 0, weight_mode = 0;
    reg [15:0] vector_len = 0;
    wire busy, done, act_rd_en, weight_rd_en;
    wire [5:0] act_rd_addr, weight_rd_addr;
    wire signed [31:0] result;
    wire [15:0] weight_word_reads;
    wire [31:0] elapsed_cycles;
    reg signed [7:0] act_rd_data;
    reg [31:0] weight_rd_data;
    reg [7:0] acts [0:63];
    reg [31:0] weights [0:15];
    int activation_reads, packed_reads, seed = 19823;
    dotprod #(.ADDR_WIDTH(6)) dut (.*);

    always @(posedge clk) begin
        if (act_rd_en) begin
            if (act_rd_addr !== activation_reads[5:0])
                $fatal(1, "Unexpected activation address %d", act_rd_addr);
            act_rd_data <= acts[act_rd_addr];
            activation_reads++;
        end
        if (weight_rd_en) begin
            if (weight_rd_addr !== packed_reads[5:0])
                $fatal(1, "Unexpected packed word address %d", weight_rd_addr);
            weight_rd_data <= weights[weight_rd_addr];
            packed_reads++;
        end
    end

    task automatic run_case(input int len, input bit w4);
        int want, terms, lanes, width, a, w, mask;
        terms = len <= 64 ? len : 0;
        lanes = w4 ? 8 : 4;
        width = w4 ? 4 : 8;
        mask = (1 << width) - 1;
        want = 0;
        activation_reads = 0;
        packed_reads = 0;
        for (int i = 0; i < 16; i++) weights[i] = 0;
        for (int i = 0; i < terms; i++) begin
            a = i == 0 ? -128 : i == 1 ? 127 : ($random(seed) & 255) - 128;
            w = i == 0 ? -(1 << (width-1)) : i == 1 ? mask/2 :
                ($random(seed) & mask) - (1 << (width-1));
            acts[i] = a;
            weights[i/lanes] |= (w & mask) << ((i%lanes)*width);
            want += a*w;
        end
        vector_len = len;
        weight_mode = w4;
        start = 1;
        @(negedge clk);
        start = 0;
        vector_len = 0;
        weight_mode = !w4;
        wait (done);
        if (busy || result !== want || activation_reads != terms ||
            packed_reads != (terms+lanes-1)/lanes ||
            weight_word_reads !== packed_reads[15:0] ||
            elapsed_cycles !== (terms == 0 ? 1 : terms+3))
            $fatal(1, "W%0d length %0d: expected sum %0d, got %0d, act reads %0d, weight reads %0d",
                   width, len, want, result, activation_reads, packed_reads);
        @(negedge clk);
    endtask

    initial begin
        repeat (3) @(negedge clk);
        rst = 0;
        for (int n = 0; n <= 65; n++) begin
            run_case(n, 0);
            run_case(n, 1);
        end
        $display("PASS 132 W8/W4 jobs, lengths 0 through 65");
        $finish;
    end
    initial begin
        #100000;
        $fatal(1, "Timeout");
    end
endmodule
