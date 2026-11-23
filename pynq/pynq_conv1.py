"""Load the first-layer overlay and compare its output with exported vectors."""

import argparse
from pathlib import Path
from time import monotonic, sleep


def read_mem(path, length):
    words = [int(line, 16) for line in path.read_text().split()]
    if len(words) != length:
        raise ValueError(f"{path}: expected {length} values, got {len(words)}")
    return words


class PynqConv1:
    def __init__(self, bitstream):
        from pynq import Overlay

        self.overlay = Overlay(str(bitstream))
        self.commands = self.overlay.commands.mmio
        self.replies = self.overlay.replies.mmio
        self.toggle = self.replies.read(8) & 1

    def command(self, op, index=0, value=None):
        if value is not None:
            self.commands.write(8, value & 0xFFFFFFFF)
        self.toggle ^= 1
        self.commands.write(0, (self.toggle << 31) | (op << 24) | index)
        deadline = monotonic() + 1
        while self.replies.read(8) & 1 != self.toggle:
            if monotonic() >= deadline:
                raise TimeoutError(f"PL did not acknowledge operation {op}")
        return self.replies.read(0)

    def run_vectors(self, directory):
        for op, name, count in ((1, "input", 3072), (2, "weights", 54), (3, "bias", 16)):
            for index, value in enumerate(read_mem(directory / (name + ".mem"), count)):
                self.command(op, index, value)
        for index, value in enumerate(read_mem(directory / "multiplier.mem", 16)):
            self.command(4, index, value)
            self.command(5, index, value >> 32)
        self.command(6)
        deadline = monotonic() + 5
        while self.replies.read(8) & 6 != 4:
            if monotonic() >= deadline:
                raise TimeoutError("PL did not finish conv1")
            sleep(0.001)
        actual = [self.command(7, index) for index in range(16384)]
        counters = {name: self.command(op) for op, name in enumerate(
            ("cycles", "image_reads", "packed_reads", "output_writes"), start=8
        )}
        return actual, counters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bitstream", type=Path)
    parser.add_argument("vectors", type=Path)
    args = parser.parse_args()
    expected = read_mem(args.vectors / "expected.mem", 16384)
    actual, counters = PynqConv1(args.bitstream).run_vectors(args.vectors)
    for index, (want, got) in enumerate(zip(expected, actual)):
        if want != got:
            channel, pixel = divmod(index, 1024)
            row, col = divmod(pixel, 32)
            raise SystemExit(f"Mismatch at CHW ({channel},{row},{col}): expected {want}, got {got}")
    print(f"PASS {len(expected)} outputs. {counters}")


if __name__ == "__main__":
    main()
