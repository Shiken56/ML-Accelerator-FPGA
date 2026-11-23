"""Generate a full conv1 fixture with signed extrema, rounding and saturation."""

import argparse
from pathlib import Path
from random import Random

from export_conv1 import pack_int4, unpack_int4, write_mem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rng = Random(207)
    image = [rng.randint(-128, 127) for _ in range(3072)]
    weights = [rng.randint(-8, 7) for _ in range(432)]
    # Constant channels cover both signs of half ties and the INT32 bias wrap.
    weights[:135] = [0] * 135
    weights[108 + 4] = 1
    image[0] = 127
    bias = [1, -1, 255, 253, (1 << 31) - 1] + [rng.randint(-5000, 5000) for _ in range(11)]
    multipliers = [1 << 30] * 5 + [rng.randint(1 << 23, 1 << 28) for _ in range(11)]
    expected = []
    for oc in range(16):
        for row in range(32):
            for col in range(32):
                accum = bias[oc]
                for ic in range(3):
                    for kr in range(3):
                        for kc in range(3):
                            r, c = row + kr - 1, col + kc - 1
                            if 0 <= r < 32 and 0 <= c < 32:
                                accum += image[ic*1024 + r*32 + c] * weights[oc*27 + ic*9 + kr*3 + kc]
                accum = (accum + (1 << 31)) % (1 << 32) - (1 << 31)
                product = accum * multipliers[oc]
                magnitude = (abs(product) + (1 << 30)) >> 31
                expected.append(min(127, max(0, magnitude if product >= 0 else -magnitude)))

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    write_mem(out / "input.mem", image, 8)
    write_mem(out / "weights.mem", pack_int4(weights), 32)
    write_mem(out / "bias.mem", bias, 32)
    write_mem(out / "multiplier.mem", multipliers, 64)
    write_mem(out / "expected.mem", expected, 8)
    words = [int(word, 16) for word in (out / "weights.mem").read_text().split()]
    if unpack_int4(words) != weights:
        raise RuntimeError("Synthetic weight round trip failed")
    print(f"Wrote {len(expected)} expected outputs and verified all {len(weights)} packed weights.")


if __name__ == "__main__":
    main()
