"""Export CIFAR-10 conv1 vectors from the existing CPU reference."""

import argparse
import json
from pathlib import Path
import sys


def write_mem(path, values, bits):
    mask = (1 << bits) - 1
    path.write_text("".join(f"{int(value) & mask:0{bits // 4}x}\n" for value in values))


def pack_int4(values):
    values = [int(value) for value in values]
    if len(values) != 432 or any(not -8 <= value <= 7 for value in values):
        raise ValueError("Expected 432 signed INT4 weights")
    return [
        sum((value & 15) << (4 * lane) for lane, value in enumerate(values[i:i+8]))
        for i in range(0, len(values), 8)
    ]


def unpack_int4(words):
    return [
        ((word >> (4 * lane)) + 8) % 16 - 8
        for word in words for lane in range(8)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image-index", type=int, default=0)
    args = parser.parse_args()
    sys.path.insert(0, str(args.reference.resolve()))
    import torch
    from torchvision import datasets, transforms
    from models.cpu_integer_resnet20 import CPUIntegerResNet20, quantize_activation

    checkpoint = args.reference / "checkpoints/mixed_precision/qat_mixed_precision_best.pt"
    model = CPUIntegerResNet20(checkpoint)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    images = datasets.CIFAR10(args.dataset, train=False, download=False, transform=transform)
    if not 0 <= args.image_index < len(images):
        parser.error("--image-index outside the CIFAR-10 test set")
    image, label = images[args.image_index]
    input_scale = model.scales["conv1"]
    output_scale = model.scales[model.order[1]]
    weight, scale, bias = model.layers["conv1"]
    if tuple(weight.shape) != (16, 3, 3, 3) or model.precision_map["conv1"] != "int4":
        raise ValueError("Unexpected conv1 shape or precision")
    with torch.inference_mode():
        integer = quantize_activation(image, input_scale)
        expected = model.convolution("conv1", image.unsqueeze(0), output_scale, 1, True)[0]
        multiplier = torch.round(input_scale * scale / output_scale * (1 << 31)).long()

    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    write_mem(out / "input.mem", integer.flatten().tolist(), 8)
    write_mem(out / "expected.mem", expected.flatten().tolist(), 8)
    flattened = weight.flatten().tolist()
    write_mem(out / "weights.mem", pack_int4(flattened), 32)
    words = [int(line, 16) for line in (out / "weights.mem").read_text().splitlines()]
    if unpack_int4(words) != flattened:
        raise RuntimeError("Packed weight round trip failed")
    write_mem(out / "bias.mem", bias.tolist(), 32)
    write_mem(out / "multiplier.mem", multiplier.tolist(), 64)
    metadata = {
        "input_shape": list(integer.shape),
        "output_shape": list(expected.shape),
        "weight_shape": list(weight.shape),
        "input_and_output_order": "channel, row, column",
        "weight_order": "output_channel, input_channel, kernel_row, kernel_column",
        "weight_packing": "eight INT4 weights per word, first weight in bits 3:0",
        "stride": 1,
        "padding": 1,
        "input_scale": float(input_scale),
        "output_scale": float(output_scale),
        "requantization_shift": 31,
        "rounding": "nearest, ties away from zero",
        "output_range_after_relu": [0, 127],
        "reference_status": "CPU reference; comparison with original GPU pending",
        "image_index": args.image_index,
        "label": label,
    }
    (out / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Exported {expected.numel()} outputs; verified all {len(flattened)} packed weights.")


if __name__ == "__main__":
    main()
