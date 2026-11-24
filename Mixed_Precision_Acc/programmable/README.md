# Programmable convolution and add overlay

The engine runs an uploaded sequence of up to 64 operations after one START. The same bitstream accepts different shapes, addresses, and weight precisions. The existing Conv1 project and overlay are the baseline.

The engine supports signed INT8 activations, signed packed INT4 or INT8 weights, binary ±1 weights, 3×3 convolution, stride 1 or 2, padding 0 or 1, per-filter bias and Q31 requantization, and optional ReLU. Scaled add combines a main tensor with a Q31-scaled second tensor. The remaining operations are stride-2 skip downsampling with centered zero channel padding, global average pooling over power-of-two spatial areas, and a linear classifier that returns signed INT32 accumulators.

Tensor data is CHW. Each filter starts on a 32-bit weight-word boundary, with the first weight in the least significant lane. Binary words store +1 as zero and −1 as one. Activation storage is 64 KiB, weight storage is 69,632 words (272 KiB), and parameter storage holds 1,024 bias/multiplier pairs. Convolution accepts up to 1,024 taps per filter. The host builds descriptors with the provided constructors, checks the supported ranges, and keeps the output span outside the live input spans.

## Build and simulate

From the repository root in a Vivado 2024.1 shell:

```powershell
vivado -mode batch -source programmable/build.tcl -log programmable/build.log -journal programmable/build.jou
xvlog -sv Mixed_Precision_Acc.srcs/sources_1/new/dotprod.v programmable/programmable_mmio.v programmable/tb_programmable.sv
xelab tb_programmable -s programmable_check
xsim programmable_check -runall
```

The build uses the original project's PS and GPIO wiring to create a separate project under `programmable/build/project`. It updates the board-ready `programmable.bit` and `programmable.hwh` in the repository's `pynq/` folder after implementation meets setup and hold timing. The default simulation checks the original 16,384 Conv1 reference values, then uploads a new program for a 3×5 INT8 stride-2 convolution followed by scaled add.

To export the trained block, use Python with CPU PyTorch and point `--reference` at the directory containing `models/cpu_integer_resnet20.py` and `checkpoints/`. The default input is your existing quantized `Mixed_precision_firstlayer/input.mem`.

```powershell
python programmable/export_block.py --reference "PATH_TO_SOFTWARE_REFERENCE" --output programmable/build/block_vectors
xsim --% programmable_check -testplusarg "block=programmable/build/block_vectors" -runall
```

The exporter runs the CPU model with the same quantized image, exports checkpoint parameters, and saves four separate expected tensors. The block simulation compares all four (65,536 values), then runs the alternate program. Conv1 starts with the image at byte address 0. The three convolution results are kept at addresses 16384, 32768, and 49152. The final residual output reuses address 0 after the image has been consumed, leaving all four results available for inspection.

### Export and simulate the full checkpoint

From the Vivado project directory, run:

```powershell
python programmable/export_full.py --reference "PATH_TO_SOFTWARE_REFERENCE" --sim-dir programmable/build/full_vectors
xvlog -sv Mixed_Precision_Acc.srcs/sources_1/new/dotprod.v programmable/programmable_mmio.v programmable/tb_full_model.sv
xelab tb_full_model -s full_model_check
xsim full_model_check -runall
```

The exporter writes `pynq/full_model.json`. The reference checkpoint and its Python CPU model are required to regenerate the file. The board uses the JSON directly. The simulator checks each operation as it finishes, before the next operation can reuse its buffer. The full program uses four 16 KiB activation banks and returns ten classifier accumulators.

## Run on the PYNQ-Z2

Upload the repository's `pynq/` folder to Jupyter on the PYNQ-Z2. Open `full_model.ipynb` to load the checkpoint once and run one START per image. The FPGA returns only ten classifier accumulators. The ARM multiplies each accumulator by its exported class scale and compares the results with the CPU reference. The notebook reports model upload, FPGA compute, and per-image ARM timings. The CLI equivalent is `python3 pynq_full_model.py` from that folder.

The prepared `pynq/cifar10_cases.json` covers test indices 0–99. Run the notebook's last cell to measure FPGA accuracy against the labels and compare all ten outputs per image with the CPU reference. To regenerate the file or use another range, use CPU PyTorch and NumPy with the extracted [CIFAR-10 Python test_batch](https://www.cs.toronto.edu/~kriz/cifar.html):

```powershell
python programmable/export_full.py --reference "PATH_TO_SOFTWARE_REFERENCE" --dataset "PATH_TO_CIFAR10_ROOT" --start 0 --count 100 --cases-output programmable/build/cifar10_cases.json
```

Copy any regenerated `cifar10_cases.json` beside the notebook. For the block and alternate regressions, see [Hardware progress](../../README.md#hardware-progress).

## Precision and weight-retention experiments

Use the same instrumented overlay for the independently trained INT8, INT4, and mixed checkpoints. The prepared files under `pynq/benchmark_data/` contain each model and its own CPU reference for CIFAR-10 test indices 0–99. Image preprocessing is shared, and input quantization uses each checkpoint's calibrated scale.

From the board's PYNQ directory, open `precision_lab.ipynb` for the live demo. Run the benchmark from a board terminal with exclusive use of the FPGA. Restart the demo kernel after terminal runs so it can load its own model and synchronize the command interface.

```bash
python3 precision_benchmark.py --limit 3 --output benchmark_smoke.json
python3 precision_benchmark.py --policy resident
python3 precision_benchmark.py --policy reload --resume
```

The full benchmark runs six groups of 100 images. The reload experiment takes longer because it sends thousands of packed words for every image. The runner writes `benchmark_results.json` after each group. Use **Reload measurements** in the comparison tab to read saved results. The resident policy uploads the model once. The reload policy uploads packed weights again before each image, with the program, biases and multipliers still resident. Policy latency includes that extra upload where applicable. Model upload is separate. Payload bytes count the 32-bit parameter and descriptor words at the application boundary. They exclude MMIO addressing and acknowledgement traffic.

The hardware tab shows a single measured inference's counters. Weight reads count accesses to local weight BRAM, including any repeated packed-word reads. Activation counters count compute-side BRAM strobes, and classifier writes count writes to the final score array. Input writes by the ARM and score reads are outside those FPGA compute counters. The latency-aware selector uses the first 70% of images for calibration and compares its recommendation and an accuracy-first baseline on the remaining images. The accuracy floor is a calibration constraint. Held-out accuracy is measured separately.

To regenerate the model and reference files for the prepared comparison, run the following from the Vivado project directory:

```powershell
foreach ($variant in "int8", "int4", "mixed") {
    python programmable/export_full.py --reference "PATH_TO_SOFTWARE_REFERENCE" --dataset "PATH_TO_TEST_BATCH" --variant $variant --output "../pynq/benchmark_data/${variant}_model.json" --cases-output "../pynq/benchmark_data/${variant}_cases.json" --sim-dir "programmable/build/benchmark_$variant"
}
```

```bash
python3 pynq_programmable.py programmable.bit block.json
python3 pynq_programmable.py programmable.bit alternate.json
```

Use `--final-only` for block output verification with fewer MMIO transfers. The cycle counter includes all operations from the single START. The ARM timings are reported separately.

## Descriptor and command format

Each descriptor has eight 32-bit words. Addresses are byte offsets for activation tensors and word offsets for weights. All output tensors must fit in the activation address space.

| Word | Fields |
|---|---|
| 0 | Opcode in bits 1:0 (1 convolution or linear, 2 add, 3 pooling or projection). Bits 2–7 are INT4, ReLU, stride 2, pad 1, binary, and linear in convolution descriptors. Bit 2 distinguishes skip projection from pooling for opcode 3. |
| 1 | Source/main address in bits 15:0, destination in 31:16. |
| 2 | Input width, height, input channels, and output channels in bytes 0–3. |
| 3 | Output width and height in bytes 0–1, or pooling's area shift in bits 19:16. |
| 4 | Packed weight-word base in bits 15:0 and 26 (high bit), and parameter base in bits 25:16. For projection, bits 7:0 hold left channel padding. |
| 5 | Add second source in bits 15:0, number of elements minus one in bits 31:16. |
| 6–7 | Add signed Q31 multiplier, low word then high word. |

The handshake uses command bit 31 as the request toggle, bits 27:24 as the opcode, and bits 15:0 as the index. Host ops 1–5 write activation, weight, bias, multiplier low, and multiplier high. Op 6 starts the program with descriptor count in the payload. Op 7 reads an activation byte, op 8 reads cycles, and op 10 writes a descriptor word. Op 11 reads a classifier accumulator and op 12 reads the full-model ABI identifier. Reply status bits 0–3 are acknowledgment, busy, done, and fault.

Weight uploads (op 2) use command bit 16 as the high address bit. ABI query op 12 at index 1 identifies the instrumented overlay. Op 13 returns cycles at the requested operation index after completion. Op 14 indexes the weight-read, activation-read, activation-write, and classifier-write counters in that order.
