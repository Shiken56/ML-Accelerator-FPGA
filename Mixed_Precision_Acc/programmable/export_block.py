"""Export Conv1 + layer1.0 for the programmable FPGA from the saved CPU model."""
import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

PROJECT = Path(__file__).resolve().parents[1]
BOARD_DIR = PROJECT.parent / "pynq"
sys.path[:0] = [str(BOARD_DIR), str(PROJECT)]
from export_conv1 import write_mem
from pynq_programmable import add, conv, pack_filters, upload_commands


def export(reference, image_mem):
    import torch
    sys.path.insert(0, str(reference.resolve()))
    from models.cpu_integer_resnet20 import CPUIntegerResNet20, quantize_activation

    checkpoint = reference / "checkpoints/mixed_precision/qat_mixed_precision_best.pt"
    model = CPUIntegerResNet20(checkpoint)
    words = [int(word, 16) for word in image_mem.read_text().split()]
    if len(words) != 3072 or any(not 0 <= word <= 255 for word in words):
        raise ValueError("Expected 3072 INT8 bytes in CHW order")
    input_scale = model.scales["conv1"]
    quantized = torch.tensor([word if word < 128 else word - 256 for word in words],
                             dtype=torch.int8).reshape(1, 3, 32, 32)
    image = quantized.float() * input_scale
    if not torch.equal(quantize_activation(image, input_scale), quantized):
        raise ValueError("Input does not round-trip through the CPU model's quantizer")

    names = ("conv1", "layer1.0.conv1", "layer1.0.conv2")
    levels = (*names, "layer1.0.add")
    bases = (16384, 32768, 49152, 0)
    with TemporaryDirectory() as traces:
        model(image, trace_dir=traces)
        expected = {
            name: {"base": base, "values": [value & 255 for value in
                   torch.load(Path(traces) / (name + ".pt"), weights_only=True).flatten().tolist()]}
            for name, base in zip(levels, bases)
        }

    weights, biases, multipliers, program = [], [], [], []
    for stage, name in enumerate(names):
        weight, weight_scale, bias = model.layers[name]
        filters, channels, _, _ = weight.shape
        source_scale = model.scales[name]
        target_scale = model.scales[model.order[model.order.index(name) + 1]]
        bits = {"int4": 4, "int8": 8}[model.precision_map[name]]
        src = 0 if stage == 0 else bases[stage - 1]
        program.append(conv(src, bases[stage], (channels, 32, 32), filters,
                            len(weights), len(biases), bits, 1, 1, stage != 2))
        weights.extend(pack_filters(weight.reshape(filters, -1).tolist(), bits))
        biases.extend(bias.tolist())
        multipliers.extend(torch.round(source_scale * weight_scale / target_scale * (1 << 31))
                           .long().tolist())

    skip_scale = model.scales[names[1]]
    output_scale = model.scales[model.order[model.order.index(names[2]) + 1]]
    program.append(add(bases[2], bases[0], bases[3], 16384,
                       round(skip_scale / output_scale * (1 << 31)), True))
    return dict(input=words, weights=weights, biases=biases, multipliers=multipliers,
                program=program, expected=expected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--input", default=Path("Mixed_precision_firstlayer/input.mem"), type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    bundle = export(args.reference, args.input)
    args.output.mkdir(parents=True, exist_ok=True)
    (BOARD_DIR / "block.json").write_text(json.dumps(bundle) + "\n")
    commands = list(upload_commands(bundle))
    write_mem(args.output / "commands.mem",
              ((op << 48) | (address << 32) | (value & 0xFFFFFFFF)
               for op, address, value in commands), 64)
    write_mem(args.output / "expected.mem",
              (((entry["base"] + i) << 8) | value for entry in bundle["expected"].values()
               for i, value in enumerate(entry["values"])), 32)
    (args.output / "counts.txt").write_text(f"{len(commands)} {len(bundle['program'])}\n")
