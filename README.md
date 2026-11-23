# ML Accelerator FPGA

A programmable mixed-precision convolution and scaled-add accelerator for the PYNQ-Z2, built with Vivado 2024.1.

- `Mixed_Precision_Acc/` contains the Vivado sources, simulation testbenches, and vector exporters.
- `pynq/` contains the board notebooks, Python drivers, reference vectors, and the matching programmable overlay files.

The board verified ResNet-20 Conv1 and the first residual block against the CPU integer reference. The same overlay also passed a separate INT8 stride-2 convolution and add test.

Upload `pynq/` to Jupyter on the PYNQ-Z2 and run `full_model.ipynb` for full ResNet-20 inference. Use `verify.ipynb` for the residual-block and INT8 regression. See the [Vivado build and simulation instructions](Mixed_Precision_Acc/programmable/README.md) to rebuild the overlay. Rebuilds update the board files in `pynq/`.
