"""Program descriptors, validation, and PYNQ runner for the programmable overlay."""
import argparse
import json
from pathlib import Path
from time import perf_counter, sleep

from pynq_conv1 import PynqConv1


def tensor(base, length):
    if length < 1 or base < 0 or base + length > 65536:
        raise ValueError("Activation tensor exceeds the 64 KiB address space")
    return range(base, base + length)


def disjoint(output, *inputs):
    if any(output.start < src.stop and src.start < output.stop for src in inputs):
        raise ValueError("Output overlaps a live input tensor")


def conv(src, dst, shape, filters, word_base, param_base, bits, stride, pad, relu):
    channels, height, width = shape
    if not all(1 <= n <= 255 for n in (*shape, filters)) or channels * 9 > 1024:
        raise ValueError("Unsupported convolution dimensions")
    if bits not in (1, 4, 8) or stride not in (1, 2) or pad not in (0, 1):
        raise ValueError("Unsupported convolution configuration")
    out_h = (height + 2 * pad - 3) // stride + 1
    out_w = (width + 2 * pad - 3) // stride + 1
    if out_h < 1 or out_w < 1:
        raise ValueError("Empty convolution output")
    disjoint(tensor(dst, filters * out_h * out_w), tensor(src, channels * height * width))
    lanes = 32 // bits
    words = filters * ((channels * 9 + lanes - 1) // lanes)
    if not (0 <= word_base <= 69632 - words and 0 <= param_base <= 1024 - filters):
        raise ValueError("Convolution parameters exceed memory capacity")
    flags = 1 | ((bits == 4) << 2) | (bool(relu) << 3) | ((stride == 2) << 4) | (pad << 5) | ((bits == 1) << 6)
    return [
        flags, src | (dst << 16),
        width | (height << 8) | (channels << 16) | (filters << 24),
        out_w | (out_h << 8), (word_base & 0xFFFF) | (param_base << 16) | ((word_base >> 16) << 26), 0, 0, 0,
    ]


def add(main, skip, dst, length, skip_multiplier, relu):
    disjoint(tensor(dst, length), tensor(main, length), tensor(skip, length))
    if not -(1 << 63) <= skip_multiplier < (1 << 63):
        raise ValueError("Skip multiplier exceeds signed 64 bits")
    return [
        2 | (bool(relu) << 3), main | (dst << 16), 0, 0, 0,
        skip | ((length - 1) << 16), skip_multiplier & 0xFFFFFFFF,
        (skip_multiplier >> 32) & 0xFFFFFFFF,
    ]


def pack_filters(filters, bits):
    if bits not in (1, 4, 8):
        raise ValueError("Unsupported weight precision")
    lanes = 32 // bits
    mask = (1 << bits) - 1
    words = []
    for weights in filters:
        if bits == 1:
            if any(weight not in (-1, 1) for weight in weights):
                raise ValueError("Binary weights must be +1 or -1")
            codes = [int(weight < 0) for weight in weights]
        else:
            if any(not -(1 << (bits - 1)) <= weight < (1 << (bits - 1)) for weight in weights):
                raise ValueError("Weight does not fit its declared precision")
            codes = [weight & mask for weight in weights]
        for offset in range(0, len(codes), lanes):
            words.append(sum(code << (bits * lane)
                             for lane, code in enumerate(codes[offset:offset + lanes])))
    return words


def stages(bundle):
    for name, entry in bundle["expected"].items():
        yield name, entry["base"], entry["values"]


def upload_commands(bundle):
    program = bundle["program"]
    if not 1 <= len(program) <= 64 or any(len(entry) != 8 for entry in program):
        raise ValueError("Expected 1–64 eight-word descriptors")
    for op, data in ((1, bundle["input"]), (2, bundle["weights"]), (3, bundle["biases"])):
        for address, value in enumerate(data):
            yield op, address, value
    for address, value in enumerate(bundle["multipliers"]):
        yield 4, address, value
        yield 5, address, value >> 32
    for pc, entry in enumerate(program):
        for word, value in enumerate(entry):
            yield 10, 8 * pc + word, value


def run(fpga, bundle, final_only=False):
    from pynq import Clocks

    commands = list(upload_commands(bundle))
    t0 = perf_counter()
    for op, address, value in commands:
        fpga.command(op, address, value)
    loaded = perf_counter()
    fpga.command(6, value=len(bundle["program"]))
    while True:
        status = fpga.replies.read(8)
        if status & 8:
            raise RuntimeError("Accelerator reported a program fault")
        if status & 6 == 4:
            break
        if perf_counter() - loaded >= 30:
            raise TimeoutError("Accelerator did not complete within 30 seconds")
        sleep(0.0005)
    done = perf_counter()
    cycles = fpga.command(8)
    read_seconds = 0.0
    failures = []
    outputs = list(stages(bundle))
    for name, base, expected in outputs[-1:] if final_only else outputs:
        before = perf_counter()
        actual = [fpga.command(7, base + i) for i in range(len(expected))]
        read_seconds += perf_counter() - before
        errors = [i for i, (reference, result) in enumerate(zip(expected, actual))
                  if reference != result]
        print(f"{'FAIL' if errors else 'PASS'} {name}: {len(expected) - len(errors):,}/{len(expected):,}")
        if errors:
            i = errors[0]
            failures.append(name)
            print(f"  First mismatch at flat index {i}: expected 0x{expected[i]:02x}, got 0x{actual[i]:02x}")
    print(f"FPGA compute: {cycles:,} cycles, {cycles / Clocks.fclk0_mhz / 1000:.3f} ms")
    print(f"ARM load: {(loaded - t0) * 1000:.3f} ms")
    print(f"ARM start to done: {(done - loaded) * 1000:.3f} ms")
    print(f"ARM verification readback: {read_seconds * 1000:.3f} ms")
    print(f"ARM load through readback: {(done - t0 + read_seconds) * 1000:.3f} ms")
    if failures:
        raise AssertionError("Mismatched stages: " + ", ".join(failures))
    return cycles


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bitstream", type=Path)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--final-only", action="store_true")
    args = parser.parse_args()
    fpga = PynqConv1(args.bitstream.resolve())
    run(fpga, json.loads(args.bundle.read_text()), final_only=args.final_only)
