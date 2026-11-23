`timescale 1ns / 1ps
module tb_full_model;
    reg clk = 0;
    always #10 clk = !clk;
    reg resetn = 0;
    reg [31:0] command = 0, payload = 0;
    wire [31:0] reply, status;
    reg toggle = 0;
    reg [63:0] commands [0:73727];
    reg [31:0] expected [0:524287];
    integer bases [0:63];
    integer lengths [0:63];
    integer manifest, scanned, commands_count, stage_count, values_count;
    integer i, stage_index, offset, checked;
    string directory;

    programmable_mmio dut (.clk(clk), .resetn(resetn), .command(command),
                           .payload(payload), .reply(reply), .status(status));

    task issue(input [3:0] op, input [15:0] address, input [31:0] data);
        begin
            @(negedge clk);
            toggle = !toggle;
            command = {toggle, 3'b0, op, 8'b0, address};
            payload = data;
            @(negedge clk);
            while (status[0] !== toggle) @(negedge clk);
        end
    endtask

    initial begin
        if (!$value$plusargs("vectors=%s", directory)) directory = "programmable/build/full_vectors";
        manifest = $fopen({directory, "/full_manifest.txt"}, "r");
        if (!manifest) $fatal(1, "Missing full-model manifest");
        scanned = $fscanf(manifest, "%d %d %d", commands_count, stage_count, values_count);
        if (scanned != 3 || commands_count < 1 || commands_count > 73728 ||
            stage_count < 1 || stage_count > 64 || values_count < 1 || values_count > 524288)
            $fatal(1, "Invalid full-model manifest");
        for (i = 0; i < stage_count; i = i + 1) begin
            scanned = $fscanf(manifest, "%d %d", bases[i], lengths[i]);
            if (scanned != 2 || lengths[i] < 1 ||
                (bases[i] >= 0 && bases[i] + lengths[i] > 65536))
                $fatal(1, "Invalid stage entry %0d", i);
        end
        $fclose(manifest);

        $readmemh({directory, "/full_commands.mem"}, commands, 0, commands_count - 1);
        $readmemh({directory, "/full_expected.mem"}, expected, 0, values_count - 1);
        repeat (5) @(negedge clk);
        resetn = 1;
        issue(12, 0, 0);
        if (reply !== 32'h4d504132) $fatal(1, "Incorrect full-model ABI");
        for (i = 0; i < commands_count; i = i + 1)
            issue(commands[i][51:48], commands[i][47:32], commands[i][31:0]);

        issue(6, 0, stage_count);
        checked = 0;
        for (stage_index = 0; stage_index < stage_count; stage_index = stage_index + 1) begin
            while (status[1] && dut.pc == stage_index) @(negedge clk);
            if (status[3]) $fatal(1, "Program fault at stage %0d", stage_index);
            for (offset = 0; offset < lengths[stage_index]; offset = offset + 1) begin
                if (bases[stage_index] >= 0) begin
                    if (dut.activations[bases[stage_index] + offset] !== expected[checked][7:0])
                        $fatal(1, "Stage %0d offset %0d: expected %h, got %h",
                               stage_index, offset, expected[checked][7:0],
                               dut.activations[bases[stage_index] + offset]);
                end else begin
                    if (!status[2]) $fatal(1, "Classifier did not finish");
                    issue(11, offset, 0);
                    if (reply !== expected[checked])
                        $fatal(1, "Classifier %0d: expected %h, got %h", offset, expected[checked], reply);
                end
                checked = checked + 1;
            end
            $display("PASS operation %0d: %0d outputs", stage_index, lengths[stage_index]);
        end
        if (checked != values_count || status[2:1] !== 2'b10)
            $fatal(1, "Incomplete full-model verification");
        issue(8, 0, 0);
        $display("PASS full model: %0d exact values over %0d operations, %0d cycles",
                 checked, stage_count, reply);
        $finish;
    end

    initial begin
        #2000000000;
        $fatal(1, "Full-model simulation timeout");
    end
endmodule
