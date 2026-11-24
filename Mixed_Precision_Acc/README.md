# Mixed-precision convolution on PYNQ-Z2

Open `Mixed_Precision_Acc.xpr` in Vivado 2024.1. The synthesis top is
`conv1_bd_wrapper`. The `conv1_bd` block design uses Global Synthesis.

## Generate the overlay in Vivado

1. Confirm `conv1_bd_wrapper` is the top under Design Sources.
2. Click **Generate Bitstream** and select one job to limit memory use.
3. Use `Mixed_Precision_Acc.runs/impl_1/conv1_bd_wrapper.bit` and
   `Mixed_Precision_Acc.gen/sources_1/bd/conv1_bd/hw_handoff/conv1_bd.hwh`
   from the same build. Give the copies on the board matching basenames.

The root-level `conv1.bit` and `conv1.hwh` are the preserved, timing-checked
overlay for the Conv1 regression.

## Source files

- `Mixed_Precision_Acc.srcs/sources_1/new/dotprod.v` contains the signed INT8
  activation and INT4/INT8 weight multiply-accumulate pipeline.
- `Mixed_Precision_Acc.srcs/sources_1/new/conv1.v` handles the first-layer
  image addresses, padding, packed filters, bias, requantization, and output.
- `Mixed_Precision_Acc.srcs/sources_1/new/conv1_mmio.v` connects the controller
  to the ARM-accessible command and reply registers and on-chip memories.
- `Mixed_Precision_Acc.srcs/sim_1/new/` contains the dot-product and convolution
  testbenches. Run `tb_conv1` from the project root so it can find
  `Mixed_precision_firstlayer/`, or pass the vector directory through the
  `vectors` simulator plusarg.

## Test on the board

Upload the matching overlay pair, `../pynq/pynq_conv1.py`, and the five `.mem` files
under `Mixed_precision_firstlayer/`. On the PYNQ-Z2, with the vector folder
next to the overlay and helper, run:

```bash
python3 pynq_conv1.py conv1.bit Mixed_precision_firstlayer
```

The test compares all 16,384 outputs and prints the PL counters. The hardware
in this project implements ResNet-20's first convolution.

See [the programmable overlay](programmable/README.md) for the separate
descriptor-driven convolution and add engine and its residual-block test.
