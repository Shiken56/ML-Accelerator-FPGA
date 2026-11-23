#!/usr/bin/env python3
"""
gen_vectors.py
Generates simulation vector files for tb_dotprod.sv:
- Synthetic 64-term test (W8 and W4)
- Corner cases: signed limits, negative result, zero inputs, W8 outside INT4
- Real first-layer test (oc=0, r=0, c=0) 27-term test for W4 and W8
"""

import os
import json
import numpy as np

OUTPUT_DIR = "sim_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def to_hex8(val):
    return f"{(val & 0xFF):02x}"

def to_hex32(val):
    return f"{(val & 0xFFFFFFFF):08x}"

# -------------------------------------------------------------
# 1. Synthetic 64-term Test Vectors
# -------------------------------------------------------------
act_pattern = [-128, 127, -3, 5]
w_pattern = [-8, 7, -1, 2]

synth_act_64 = act_pattern * 16  # 64 elements
synth_w_64 = w_pattern * 16      # 64 elements

# Verify product sum
expected_synth_result = sum(a * w for a, w in zip(synth_act_64, synth_w_64))
assert expected_synth_result == 30816, f"Expected 30816, got {expected_synth_result}"

# Write synthetic activations (8-bit hex per line)
with open(os.path.join(OUTPUT_DIR, "synth_act_64.mem"), "w") as f:
    for a in synth_act_64:
        f.write(f"{to_hex8(a)}\n")

# Pack W8 weights into 32-bit words (4 bytes per word, first byte in bits 7:0)
synth_w8_words = []
for i in range(0, 64, 4):
    w0 = synth_w_64[i] & 0xFF
    w1 = synth_w_64[i+1] & 0xFF
    w2 = synth_w_64[i+2] & 0xFF
    w3 = synth_w_64[i+3] & 0xFF
    word = (w3 << 24) | (w2 << 16) | (w1 << 8) | w0
    synth_w8_words.append(word)

assert len(synth_w8_words) == 16
assert f"{synth_w8_words[0]:08X}" == "02FF07F8", f"Got {synth_w8_words[0]:08X}"

with open(os.path.join(OUTPUT_DIR, "synth_w8_64.mem"), "w") as f:
    for word in synth_w8_words:
        f.write(f"{to_hex32(word)}\n")

# Pack W4 weights into 32-bit words (8 nibbles per word, first nibble in bits 3:0)
synth_w4_words = []
for i in range(0, 64, 8):
    word = 0
    for n in range(8):
        nibble = synth_w_64[i + n] & 0x0F
        word |= (nibble << (4 * n))
    synth_w4_words.append(word)

assert len(synth_w4_words) == 8
assert f"{synth_w4_words[0]:08X}" == "2F782F78", f"Got {synth_w4_words[0]:08X}"

with open(os.path.join(OUTPUT_DIR, "synth_w4_64.mem"), "w") as f:
    for word in synth_w4_words:
        f.write(f"{to_hex32(word)}\n")

print(f"Synthetic 64-term test: Expected result = {expected_synth_result}")
print(f"  First W8 word: {synth_w8_words[0]:08X} (Total words: {len(synth_w8_words)})")
print(f"  First W4 word: {synth_w4_words[0]:08X} (Total words: {len(synth_w4_words)})")

# -------------------------------------------------------------
# 2. Corner Cases (64-term tests)
# -------------------------------------------------------------
# Case A: Signed Limits (Max positive product)
# act = -128, w = -8 (W4) or -128 (W8)
# Let's do 64 terms of act = -128, w = 7 (W4 max pos) -> -128 * 7 = -896 * 64 = -57344
# Or act = -128, w = -8 -> 1024 * 64 = 65536
corner_act_limits = [-128] * 64
corner_w_limits = [-8] * 64
expected_limits_result = 65536

with open(os.path.join(OUTPUT_DIR, "corner_act_limits.mem"), "w") as f:
    for a in corner_act_limits:
        f.write(f"{to_hex8(a)}\n")

# W8 packed
with open(os.path.join(OUTPUT_DIR, "corner_w8_limits.mem"), "w") as f:
    for i in range(0, 64, 4):
        word = sum((corner_w_limits[i+k] & 0xFF) << (8*k) for k in range(4))
        f.write(f"{to_hex32(word)}\n")

# W4 packed
with open(os.path.join(OUTPUT_DIR, "corner_w4_limits.mem"), "w") as f:
    for i in range(0, 64, 8):
        word = sum((corner_w_limits[i+k] & 0x0F) << (4*k) for k in range(8))
        f.write(f"{to_hex32(word)}\n")

# Case B: All Zeros
with open(os.path.join(OUTPUT_DIR, "corner_act_zero.mem"), "w") as f:
    for _ in range(64):
        f.write("00\n")
with open(os.path.join(OUTPUT_DIR, "corner_w8_zero.mem"), "w") as f:
    for _ in range(16):
        f.write("00000000\n")
with open(os.path.join(OUTPUT_DIR, "corner_w4_zero.mem"), "w") as f:
    for _ in range(8):
        f.write("00000000\n")

# Case C: Negative Result
# act = 100, w = -5 -> sum = -500 * 64 = -32000
corner_act_neg = [100] * 64
corner_w_neg = [-5] * 64
expected_neg_result = -32000

with open(os.path.join(OUTPUT_DIR, "corner_act_neg.mem"), "w") as f:
    for a in corner_act_neg:
        f.write(f"{to_hex8(a)}\n")

with open(os.path.join(OUTPUT_DIR, "corner_w8_neg.mem"), "w") as f:
    for i in range(0, 64, 4):
        word = sum((corner_w_neg[i+k] & 0xFF) << (8*k) for k in range(4))
        f.write(f"{to_hex32(word)}\n")

with open(os.path.join(OUTPUT_DIR, "corner_w4_neg.mem"), "w") as f:
    for i in range(0, 64, 8):
        word = sum((corner_w_neg[i+k] & 0x0F) << (4*k) for k in range(8))
        f.write(f"{to_hex32(word)}\n")

# Case D: W8 with weights outside INT4 range (e.g. w = -100, +120)
# act = [-10, 20, -30, 40] * 16
# w = [-100, 120, -80, 90] * 16
# 1 period: (-10)*(-100) + 20*120 + (-30)*(-80) + 40*90 = 1000 + 2400 + 2400 + 3600 = 9400
# 16 periods = 150400
corner_act_w8large = [-10, 20, -30, 40] * 16
corner_w8large = [-100, 120, -80, 90] * 16
expected_w8large_result = sum(a * w for a, w in zip(corner_act_w8large, corner_w8large))
assert expected_w8large_result == 150400

with open(os.path.join(OUTPUT_DIR, "corner_act_w8large.mem"), "w") as f:
    for a in corner_act_w8large:
        f.write(f"{to_hex8(a)}\n")

with open(os.path.join(OUTPUT_DIR, "corner_w8large.mem"), "w") as f:
    for i in range(0, 64, 4):
        word = sum((corner_w8large[i+k] & 0xFF) << (8*k) for k in range(4))
        f.write(f"{to_hex32(word)}\n")

# -------------------------------------------------------------
# 3. Real Data Case: First Layer (oc=0, r=0, c=0)
# -------------------------------------------------------------
firstlayer_dir = "Mixed_precision_firstlayer"
inp_npy = np.load(os.path.join(firstlayer_dir, "input.npy"))[0] # (3, 32, 32)
exp_npy = np.load(os.path.join(firstlayer_dir, "expected.npy"))[0] # (16, 32, 32)

with open(os.path.join(firstlayer_dir, "weights.mem")) as f:
    weight_words = [int(line.strip(), 16) for line in f if line.strip()]

weights_unpacked = []
for word in weight_words:
    for i in range(8):
        nibble = (word >> (4 * i)) & 0xF
        if nibble >= 8:
            nibble -= 16
        weights_unpacked.append(nibble)

weights_arr = np.array(weights_unpacked, dtype=np.int32).reshape((16, 3, 3, 3))
inp_pad = np.pad(inp_npy, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)

# Extract patch for oc=0, r=0, c=0
patch_00 = inp_pad[:, 0:3, 0:3] # (3, 3, 3)
w_filter_00 = weights_arr[0]    # (3, 3, 3)

# Flatten in channel, row, column order (as specified in config.json)
real_act_27 = patch_00.flatten().tolist()
real_w_27 = w_filter_00.flatten().tolist()

real_raw_dot = sum(a * w for a, w in zip(real_act_27, real_w_27))
assert real_raw_dot == 38, f"Expected 38, got {real_raw_dot}"

# Write real activations (27 lines, 8-bit hex)
with open(os.path.join(OUTPUT_DIR, "real_act_27.mem"), "w") as f:
    for a in real_act_27:
        f.write(f"{to_hex8(a)}\n")

# Pack real W4 weights into 32-bit words
# 27 weights / 8 = 3 full words (24 weights) + 1 partial word (3 weights) = 4 words
real_w4_words = []
for word_idx in range(4):
    word = 0
    for n in range(8):
        idx = word_idx * 8 + n
        nibble = (real_w_27[idx] & 0x0F) if idx < 27 else 0
        word |= (nibble << (4 * n))
    real_w4_words.append(word)

with open(os.path.join(OUTPUT_DIR, "real_w4_27.mem"), "w") as f:
    for word in real_w4_words:
        f.write(f"{to_hex32(word)}\n")

# Pack real W8 weights (sign-extended) into 32-bit words
# 27 weights / 4 = 6 full words (24 weights) + 1 partial word (3 weights) = 7 words
real_w8_words = []
for word_idx in range(7):
    word = 0
    for b in range(4):
        idx = word_idx * 4 + b
        byte = (real_w_27[idx] & 0xFF) if idx < 27 else 0
        word |= (byte << (8 * b))
    real_w8_words.append(word)

with open(os.path.join(OUTPUT_DIR, "real_w8_27.mem"), "w") as f:
    for word in real_w8_words:
        f.write(f"{to_hex32(word)}\n")

print(f"Real data (oc=0, r=0, c=0): Length = 27, Expected Raw Result = {real_raw_dot}")
print(f"  Real W4 words: {len(real_w4_words)} words")
print(f"  Real W8 words: {len(real_w8_words)} words")

summary = {
    "expected_synth_result": expected_synth_result,
    "expected_limits_result": expected_limits_result,
    "expected_neg_result": expected_neg_result,
    "expected_w8large_result": expected_w8large_result,
    "real_raw_dot": real_raw_dot,
    "real_len": 27
}

with open(os.path.join(OUTPUT_DIR, "expected_results.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("Vector generation complete! Saved in sim_data/")
