# Programmable convolution and add overlay

The engine runs an uploaded sequence of up to 64 operations after one START. The same bitstream accepts different shapes, addresses, and weight precisions. The existing Conv1 project and overlay are the baseline.

The engine supports signed INT8 activations, signed packed INT4 or INT8 weights, 3×3 convolution, stride 1 or 2, padding 0 or 1, per-filter bias and Q31 requantization, and optional ReLU. The add engine combines a main tensor with a Q31-scaled second tensor before rounding and saturation. Tensor data is CHW. Each convolution filter starts on a 32-bit weight-word boundary, with the first weight in the least significant lane. Activation storage is 64 KiB, weight storage is 4096 words, and parameter storage holds 256 bias/multiplier pairs. Convolution accepts up to 1024 taps per filter. The host builds descriptors with the provided constructors, checks the supported ranges, and keeps the output span outside the live input spans.

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

## Run on the PYNQ-Z2

Upload the repository's `pynq/` folder to Jupyter on the PYNQ-Z2. Open `verify.ipynb` and run its cells. The exporter updates `pynq/block.json` directly. The block case should print four PASS lines of 16,384 values each. The alternate case should print two PASS lines of two values each. A terminal run is also supported.

```bash
python3 pynq_programmable.py programmable.bit block.json
python3 pynq_programmable.py programmable.bit alternate.json
```

Use `--final-only` for block output verification with fewer MMIO transfers. The cycle counter includes all operations from the single START. The ARM timings are reported separately.

## Descriptor and command format

Each descriptor has eight 32-bit words. Addresses are byte offsets for activation tensors and word offsets for weights. All output tensors must fit in the activation address space.

| Word | Fields |
|---|---|
| 0 | Opcode in bits 1:0 (1 convolution, 2 add). Bits 2–5 are INT4, ReLU, stride 2, and pad 1, respectively. |
| 1 | Source/main address in bits 15:0, destination in 31:16. |
| 2 | Convolution width, height, input channels, output channels in bytes 0–3. |
| 3 | Convolution output width and height in bytes 0–1. |
| 4 | Packed weight-word base in bits 15:0, bias/multiplier base in bits 23:16. |
| 5 | Add second source in bits 15:0, number of elements minus one in bits 31:16. |
| 6–7 | Add signed Q31 multiplier, low word then high word. |

The handshake uses command bit 31 as the request toggle, bits 27:24 as the opcode, and bits 15:0 as the index. Host ops 1–5 write activation, weight, bias, multiplier low, and multiplier high. Op 6 starts the program with descriptor count in the payload. Op 7 reads an activation byte, op 8 reads cycles, and op 10 writes a descriptor word. Reply status bits 0–3 are acknowledgment, busy, done, and fault.
