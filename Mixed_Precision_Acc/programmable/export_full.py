"""Export the full ResNet-20 checkpoint and CPU reference for the PYNQ accelerator."""
import argparse
from hashlib import md5
import json
from pathlib import Path
import pickle
import sys
from tempfile import TemporaryDirectory

PROJECT = Path(__file__).resolve().parents[1]
BOARD_DIR = PROJECT.parent / "pynq"
sys.path[:0] = [str(BOARD_DIR), str(PROJECT)]
from export_conv1 import write_mem
from pynq_full_model import linear, pool, project
from pynq_programmable import add, conv, pack_filters, upload_commands

BITS = {"int8": 8, "int4": 4, "bnn": 1}
BANKS = (0, 16384, 32768, 49152)


def load_reference(reference):
    sys.path.insert(0, str(reference.resolve()))
    from models.cpu_integer_resnet20 import CPUIntegerResNet20, quantize_activation

    checkpoint = reference / "checkpoints/mixed_precision/qat_mixed_precision_best.pt"
    return CPUIntegerResNet20(checkpoint), quantize_activation


def export_program(model):
    import torch

    weights, biases, multipliers, program, trace = [], [], [], [], []
    precision = {}
    scales = model.scales

    def emit(descriptor, name, destination, shortcut=None):
        program.append(descriptor)
        trace.append((name, destination, shortcut))

    def convolution(name, source, destination, shape, stride, relu):
        kernel, weight_scale, bias = model.layers[name]
        filters, channels, h, w = kernel.shape
        if channels != shape[0] or (h, w) != (3, 3):
            raise ValueError(f"Unexpected shape for {name}")
        bits = precision[name] = BITS[model.precision_map[name]]
        target = model.order[model.order.index(name) + 1]
        emit(conv(source, destination, shape, filters, len(weights), len(biases),
                  bits, stride, 1, relu), name, destination)
        weights.extend(pack_filters(kernel.reshape(filters, -1).tolist(), bits))
        biases.extend(bias.tolist())
        multipliers.extend(torch.round(scales[name] * weight_scale / scales[target] * (1 << 31))
                           .long().tolist())
        return filters, (shape[1] + stride - 1) // stride, (shape[2] + stride - 1) // stride

    source = BANKS[1]
    shape = convolution("conv1", BANKS[0], source, (3, 32, 32), 1, True)
    for stage in range(1, 4):
        for block in range(3):
            prefix = f"layer{stage}.{block}"
            first, second = prefix + ".conv1", prefix + ".conv2"
            first_dst, second_dst, skip_dst = (addr for addr in BANKS if addr != source)
            stride = 2 if stage > 1 and block == 0 else 1
            next_shape = convolution(first, source, first_dst, shape, stride, True)
            if convolution(second, first_dst, second_dst, next_shape, 1, False) != next_shape:
                raise ValueError(f"Incompatible residual shape for {prefix}")
            output_scale = scales[model.order[model.order.index(second) + 1]]
            skip_multiplier = round(scales[first] / output_scale * (1 << 31))
            skip_source = source
            if stride == 2:
                prev = f"layer{stage - 1}.2.add"
                emit(project(source, skip_dst, shape, next_shape[0]), prefix + ".skip", skip_dst,
                     (prev, shape[0], next_shape[0]))
                skip_source = skip_dst
            channels, height, width = next_shape
            emit(add(second_dst, skip_source, first_dst, channels * height * width,
                     skip_multiplier, True), prefix + ".add", first_dst)
            source, shape = first_dst, next_shape

    pool_dst = next(addr for addr in BANKS if addr != source)
    emit(pool(source, pool_dst, shape), "pool", pool_dst)
    kernel, weight_scale, bias = model.layers["linear"]
    outputs, inputs = kernel.shape
    if inputs != shape[0] or outputs != 10:
        raise ValueError("Unexpected CIFAR-10 classifier shape")
    bits = precision["linear"] = BITS[model.precision_map["linear"]]
    emit(linear(pool_dst, inputs, outputs, len(weights), len(biases), bits), "linear_accum", None)
    weights.extend(pack_filters(kernel.tolist(), bits))
    biases.extend(bias.tolist())
    classifier_scales = (scales["linear"] * weight_scale).float().tolist()
    return dict(input=[], weights=weights, biases=biases, multipliers=multipliers, program=program,
                classifier_scales=classifier_scales, precision=precision), trace


def cpu_case(model, image, image_words, trace_dir, name, label=None):
    import torch

    scores = model(image, trace_dir=trace_dir)
    accum = torch.load(Path(trace_dir) / "linear_accum.pt", weights_only=True).flatten()
    return dict(name=name, input=image_words, accum=accum.tolist(), scores=scores.flatten().tolist(),
                label=label)


def write_simulation(directory, model, case, stages, traces):
    import torch
    import torch.nn.functional as F

    directory.mkdir(parents=True, exist_ok=True)
    commands = list(upload_commands(dict(model, input=case["input"])))
    write_mem(directory / "full_commands.mem",
              ((op << 48) | (address << 32) | (data & 0xFFFFFFFF)
               for op, address, data in commands), 64)
    expected, manifest = [], []
    for name, base, shortcut in stages:
        if shortcut:
            previous, old_channels, new_channels = shortcut
            values = torch.load(Path(traces) / (previous + ".pt"), weights_only=True)
            difference = new_channels - old_channels
            left = difference // 2
            values = F.pad(values[:, :, ::2, ::2], (0, 0, 0, 0, left, difference - left))
        else:
            values = torch.load(Path(traces) / (name + ".pt"), weights_only=True)
        manifest.append(f"{-1 if base is None else base} {values.numel()}\n")
        expected.extend(values.flatten().tolist())
    write_mem(directory / "full_expected.mem", expected, 32)
    (directory / "full_manifest.txt").write_text(
        f"{len(commands)} {len(stages)} {len(expected)}\n" + "".join(manifest))


def write_labeled_cases(model, quantize, dataset, start, count, output):
    import torch

    if dataset.is_dir():
        batch = dataset / "test_batch"
        dataset = batch if batch.is_file() else dataset / "cifar-10-batches-py/test_batch"
    contents = dataset.read_bytes()
    if md5(contents).hexdigest() != "40351d587109b95175f43aff81a1287e":
        raise ValueError("Expected the original CIFAR-10 Python test_batch")
    # The archive contains a NumPy array and labels in official test order.
    batch = pickle.loads(contents, encoding="bytes")
    if not (0 <= start < start + count <= len(batch[b"labels"])):
        raise ValueError("Requested indices are outside the CIFAR-10 test set")
    mean = torch.tensor((0.4914, 0.4822, 0.4465)).view(3, 1, 1)
    std = torch.tensor((0.2470, 0.2435, 0.2616)).view(3, 1, 1)
    cases = []
    with TemporaryDirectory() as traces:
        for first in range(start, start + count, 8):
            stop = min(first + 8, start + count)
            images = torch.from_numpy(batch[b"data"][first:stop]).reshape(-1, 3, 32, 32).float()
            images.div_(255).sub_(mean).div_(std)
            quantized = quantize(images, model.scales["conv1"]).flatten(1).tolist()
            scores = model(images, trace_dir=traces).tolist()
            accum = torch.load(Path(traces) / "linear_accum.pt", weights_only=True).tolist()
            for offset, index in enumerate(range(first, stop)):
                cases.append(dict(name=f"test[{index}]", label=batch[b"labels"][index],
                                  input=[value & 255 for value in quantized[offset]],
                                  accum=accum[offset], scores=scores[offset]))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(split=f"CIFAR-10 test indices {start}–{start + count - 1}",
                                      cases=cases)) + "\n", encoding="utf-8")


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--input", default=PROJECT / "Mixed_precision_firstlayer/input.mem", type=Path)
    parser.add_argument("--output", default=BOARD_DIR / "full_model.json", type=Path)
    parser.add_argument("--sim-dir", type=Path)
    parser.add_argument("--dataset", type=Path, help="Official CIFAR-10 test_batch or containing directory")
    parser.add_argument("--cases-output", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()
    if bool(args.dataset) != bool(args.cases_output):
        parser.error("--dataset and --cases-output must be supplied together")

    model, quantize = load_reference(args.reference)
    bundle, stages = export_program(model)
    words = [int(word, 16) for word in args.input.read_text().split()]
    if len(words) != 3072 or any(not 0 <= word <= 255 for word in words):
        raise ValueError("Expected 3072 INT8 bytes in CHW order")
    quantized = torch.tensor([x if x < 128 else x - 256 for x in words],
                             dtype=torch.int8).reshape(1, 3, 32, 32)
    input_scale = model.scales["conv1"]
    image = quantized.float() * input_scale
    if not torch.equal(quantize(image, input_scale), quantized):
        raise ValueError("Input does not round-trip through the CPU quantizer")

    with TemporaryDirectory() as traces:
        bundle["sample"] = cpu_case(model, image, words, traces, "exported Conv1 image")
        if args.sim_dir:
            write_simulation(args.sim_dir, bundle, bundle["sample"], stages, traces)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bundle) + "\n", encoding="utf-8")
    print(f"Exported {len(bundle['program'])} operations, {len(bundle['weights']):,} packed weight words, "
          f"{len(bundle['biases'])} biases; reference class {max(range(10), key=bundle['sample']['scores'].__getitem__)}")
    if args.dataset:
        write_labeled_cases(model, quantize, args.dataset, args.start, args.count, args.cases_output)


if __name__ == "__main__":
    main()
