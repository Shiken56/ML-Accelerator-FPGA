# ML Accelerator FPGA

A programmable ResNet-20 accelerator for PYNQ-Z2, built in Vivado 2024.1. It runs the entire network with one START per image and supports INT8, INT4, and binary weights with INT8 activations. Parameters are loaded once and intermediate tensors stay in FPGA memory.

## Hardware progress

| Milestone | How to verify it | Evidence |
| --- | --- | --- |
| First convolution | Run the [Conv1 regression](Mixed_Precision_Acc/README.md#test-on-the-board). | All 16,384 outputs matched on the board. |
| First residual block and configurable operations | From the board's `pynq/` directory, run `python3 pynq_programmable.py programmable.bit block.json`, then `python3 pynq_programmable.py programmable.bit alternate.json`. | The four block tensors matched on the board. The alternate INT8 stride-2 convolution and scaled add passed on the same overlay. |
| Full mixed-precision ResNet-20 | Run [full_model.ipynb](pynq/full_model.ipynb). | The board matched the CPU integer reference on 100/100 images. Both classified 92/100 correctly. |
| Three-way precision comparison | Open [precision_lab.ipynb](pynq/precision_lab.ipynb) and run the benchmark below. | All three configurations passed RTL simulation. Use the same instrumented overlay to collect the board comparison. |

## Board notebooks

Copy the entire `pynq/` folder to the board, including `benchmark_data/`. Start Jupyter in that folder. Keep `programmable.bit` and its matching `programmable.hwh` beside the notebooks.

- [full_model.ipynb](pynq/full_model.ipynb) verifies the complete mixed-precision network. Run the setup cell to program the FPGA and upload the model. Run the single-image cell to compare the ten classifier outputs with the saved CPU reference. Run the optional evaluation cell for CIFAR-10 test indices 0-99. The 100-image run took about two minutes on the earlier board baseline.
- [precision_lab.ipynb](pynq/precision_lab.ipynb) is the live demo and instrumented hardware check. Run its code cell, choose a weight configuration and image index, and click **Run on FPGA**. **Live inference** compares the classifier scores and shows the prediction. **Hardware execution** shows per-operation cycles and local memory accesses for that image. **Precision comparison** shows the CPU references until board measurements are available.

To collect board measurements, close any notebook session using the overlay and run these commands from the board's `pynq/` directory:

```bash
python3 precision_benchmark.py --policy resident
python3 precision_benchmark.py --policy reload --resume
```

The first command keeps weights resident across images. The second reloads weights before each image and appends those measurements. Both use CIFAR-10 test indices 0-99 and each checkpoint's own CPU reference. Open or restart the demo notebook and click **Reload measurements** to read `benchmark_results.json`.

## Sources and builds

`Mixed_Precision_Acc/` contains the Vivado project, RTL, testbenches, and exporters. See the [programmable overlay build and simulation instructions](Mixed_Precision_Acc/programmable/README.md) to rebuild the overlay or regenerate the benchmark data.
