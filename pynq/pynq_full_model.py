"""Full-model descriptors and one-START inference for the PYNQ-Z2."""
import json
import struct
from pathlib import Path
from time import perf_counter, sleep

from pynq_conv1 import PynqConv1
from pynq_programmable import disjoint, tensor, upload_commands

ABI = 0x4D504132
CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


def project(src, dst, shape, filters):
    channels, h, w = shape
    if not (1 <= channels <= filters <= 255 and 1 <= h <= 255 and 1 <= w <= 255):
        raise ValueError("Unsupported skip projection shape")
    out_h, out_w = (h + 1) // 2, (w + 1) // 2
    disjoint(tensor(dst, filters * out_h * out_w), tensor(src, channels * h * w))
    return [7, src | dst << 16, w | h << 8 | channels << 16 | filters << 24,
            out_w | out_h << 8, (filters - channels) // 2, 0, 0, 0]


def pool(src, dst, shape):
    channels, h, w = shape
    area = h * w
    if not (1 <= channels <= 255 and 1 <= h <= 255 and 1 <= w <= 255):
        raise ValueError("Unsupported pooling shape")
    if area & (area - 1):
        raise ValueError("Pooling plane must have a power-of-two number of pixels")
    disjoint(tensor(dst, channels), tensor(src, channels * area))
    return [3, src | dst << 16, w | h << 8 | channels << 16 | channels << 24,
            (area.bit_length() - 1) << 16, 0, 0, 0, 0]


def linear(src, inputs, outputs, word_base, param_base, bits):
    tensor(src, inputs)
    if not (1 <= inputs <= 255 and 1 <= outputs <= 255 and bits in (1, 4, 8)):
        raise ValueError("Unsupported classifier configuration")
    lanes = 32 // bits
    words = outputs * ((inputs + lanes - 1) // lanes)
    if not (0 <= word_base <= 65536 - words and 0 <= param_base <= 1024 - outputs):
        raise ValueError("Classifier parameters exceed memory capacity")
    flags = 129 | ((bits == 4) << 2) | ((bits == 1) << 6)
    return [flags, src, 0x0101 | inputs << 16 | outputs << 24, 0x0101,
            word_base | param_base << 16, 0, 0, 0]


def float32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


class FullModel:
    def __init__(self, overlay, model):
        from pynq import Clocks

        self.fpga = PynqConv1(Path(overlay).resolve())
        if self.fpga.command(12) != ABI:
            raise RuntimeError("This bitstream does not support the full-model instruction set")
        self.clock_mhz = Clocks.fclk0_mhz
        self.operation_count = len(model["program"])
        self.scales = model["classifier_scales"]
        if len(self.scales) != len(CLASSES):
            raise ValueError("Expected ten CIFAR-10 classifier scales")
        started = perf_counter()
        for op, address, data in upload_commands(model):
            self.fpga.command(op, address, data)
        self.model_load_ms = (perf_counter() - started) * 1000

    def infer(self, image):
        if len(image) != 3072 or any(not 0 <= byte <= 255 for byte in image):
            raise ValueError("Expected 3072 quantized image bytes in CHW order")
        started = perf_counter()
        for address, byte in enumerate(image):
            self.fpga.command(1, address, byte)
        loaded = perf_counter()
        self.fpga.command(6, value=self.operation_count)
        while True:
            status = self.fpga.replies.read(8)
            if status & 8:
                raise RuntimeError("Full-model program fault")
            if status & 6 == 4:
                break
            if perf_counter() - loaded >= 30:
                raise TimeoutError("Full-model accelerator did not complete")
            sleep(0.0005)
        finished = perf_counter()
        raw = [self.fpga.command(11, index) for index in range(len(CLASSES))]
        read = perf_counter()
        accum = [value if value < 2**31 else value - 2**32 for value in raw]
        scores = [float32(float32(value) * scale) for value, scale in zip(accum, self.scales)]
        cycles = self.fpga.command(8)
        return {
            "accum": accum, "scores": scores, "prediction": max(range(len(scores)), key=scores.__getitem__),
            "cycles": cycles, "compute_ms": cycles / self.clock_mhz / 1000,
            "image_load_ms": (loaded - started) * 1000, "start_to_done_ms": (finished - loaded) * 1000,
            "readback_ms": (read - finished) * 1000, "total_ms": (read - started) * 1000,
        }


def verify(engine, cases):
    agreements = correct = reference_correct = labeled = 0
    compute_total = wall_total = 0.0
    for case in cases:
        result = engine.infer(case["input"])
        matches = result["accum"] == case["accum"] and result["scores"] == case["scores"]
        agreements += matches
        if not matches:
            print(f"FAIL {case['name']}: CPU {case['accum']}, FPGA {result['accum']}")
        if case.get("label") is not None:
            expected_class = max(range(len(CLASSES)), key=case["scores"].__getitem__)
            correct += result["prediction"] == case["label"]
            reference_correct += expected_class == case["label"]
            labeled += 1
        compute_total += result["compute_ms"]
        wall_total += result["total_ms"]
        if len(cases) == 1:
            print(f"Predicted class: {CLASSES[result['prediction']]} ({result['prediction']})")
            print(f"Scores: {result['scores']}")
            print(f"FPGA: {result['cycles']:,} cycles, {result['compute_ms']:.3f} ms")
            print(f"Image upload: {result['image_load_ms']:.3f} ms")
            print(f"START to done: {result['start_to_done_ms']:.3f} ms")
            print(f"Ten-score readback: {result['readback_ms']:.3f} ms")
            print(f"Image upload through scores: {result['total_ms']:.3f} ms")
    if not cases:
        raise ValueError("No verification cases")
    print(f"{'PASS' if agreements == len(cases) else 'FAIL'}: {agreements}/{len(cases)} images match the CPU reference")
    if labeled:
        print(f"FPGA accuracy: {correct}/{labeled} ({correct / labeled:.1%})")
        print(f"CPU reference accuracy: {reference_correct}/{labeled} ({reference_correct / labeled:.1%})")
    print(f"Mean FPGA compute: {compute_total / len(cases):.3f} ms")
    print(f"Mean ARM image upload through scores: {wall_total / len(cases):.3f} ms")
    if agreements != len(cases):
        raise AssertionError("Full-model CPU/FPGA mismatch")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", default=Path("programmable.bit"), type=Path)
    parser.add_argument("--model", default=Path("full_model.json"), type=Path)
    parser.add_argument("--cases", type=Path)
    args = parser.parse_args()
    model = json.loads(args.model.read_text())
    engine = FullModel(args.overlay, model)
    print(f"One-time model upload: {engine.model_load_ms:.3f} ms")
    cases = json.loads(args.cases.read_text())["cases"] if args.cases else [model["sample"]]
    verify(engine, cases)
