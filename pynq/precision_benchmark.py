"""Same-overlay precision, timing, and weight-retention experiments."""
from hashlib import sha256
import json
import math
import statistics
from pathlib import Path
from time import perf_counter

from pynq_full_model import FullModel

VARIANTS = ("int8", "int4", "mixed")
POLICIES = ("resident", "reload")
COLORS = {"int8": "#2878C8", "int4": "#7655B4", "mixed": "#1C8B77", "bnn": "#DC791C"}
COUNTERS = ("weight_reads", "activation_reads", "activation_writes", "classifier_writes")


def load_suite(directory=Path("benchmark_data")):
    suite = {}
    reference = None
    for name in VARIANTS:
        model = json.loads((directory / f"{name}_model.json").read_text())
        dataset = json.loads((directory / f"{name}_cases.json").read_text())
        identity = (dataset["split"], [(c["name"], c["label"]) for c in dataset["cases"]])
        if reference is not None and identity != reference:
            raise ValueError("Variants must use identical image indices and labels")
        reference = identity
        if name != "mixed" and set(model["precision"].values()) != {int(name[-1])}:
            raise ValueError(f"Unexpected weight precision in {name}")
        suite[name] = model, dataset["cases"]
    return suite


def read_counters(engine, model):
    fpga = engine.fpga
    stages = [(name, fpga.command(13, i)) for i, name in enumerate(model["operations"])]
    memory = {name: fpga.command(14, i) for i, name in enumerate(COUNTERS)}
    if sum(value for _, value in stages) != fpga.command(8) or memory["classifier_writes"] != 10:
        raise RuntimeError("Invalid operation or memory counters")
    return stages, memory


def exact_count(case, result):
    return sum(a == b and c == d for a, b, c, d in
               zip(case["accum"], result["accum"], case["scores"], result["scores"]))


def run_suite(suite, output=Path("benchmark_results.json"), limit=None,
              policies=POLICIES, resume=False):
    groups = json.loads(output.read_text()) if resume and output.exists() else []
    for variant, (model, cases) in suite.items():
        cases = cases[:limit]
        if not cases:
            raise ValueError("Benchmark requires at least one image")
        fingerprint = sha256(json.dumps([model, cases], sort_keys=True).encode()).hexdigest()
        previous = [g for g in groups if g["variant"] == variant]
        if any(g.get("fingerprint") != fingerprint for g in previous):
            raise ValueError(f"{variant}: existing results use different inputs or model")
        pending = [policy for policy in policies if policy not in {g["policy"] for g in previous}]
        if not pending:
            continue
        engine = FullModel("programmable.bit", model, profile=True)
        if previous and any(g["clock_mhz"] != engine.clock_mhz for g in previous):
            raise ValueError(f"{variant}: existing results use a different FPGA clock")
        weight_bytes = 4 * len(model["weights"])
        parameter_bytes = weight_bytes + 4 * len(model["biases"]) + 8 * len(model["multipliers"])
        program_bytes = 32 * len(model["program"])
        warmup = engine.infer(cases[0]["input"])
        if exact_count(cases[0], warmup) != 10:
            raise AssertionError(f"{variant} warm-up mismatch")
        stages, memory = read_counters(engine, model)
        for policy in pending:
            rows = []
            for i, case in enumerate(cases):
                reload_ms = 0.0
                if policy == "reload":
                    start = perf_counter()
                    for address, word in enumerate(model["weights"]):
                        engine.fpga.command(2, address, word)
                    reload_ms = (perf_counter() - start) * 1000
                result = engine.infer(case["input"])
                hits = exact_count(case, result)
                if hits != 10:
                    raise AssertionError(f"{variant} {policy} {case['name']}: {hits}/10 exact outputs")
                if result["cycles"] != warmup["cycles"]:
                    raise AssertionError("Unexpected change in hardware cycle count")
                rows.append({key: result[key] for key in
                             ("cycles", "compute_ms", "image_load_ms", "start_to_done_ms", "readback_ms",
                              "score_scale_ms", "total_ms", "prediction")})
                rows[-1].update(image=case["name"], label=case["label"], exact_outputs=hits,
                                reload_ms=reload_ms, policy_ms=result["total_ms"] + reload_ms,
                                control_ms=result["start_to_done_ms"] - result["compute_ms"])
                if (i + 1) % 10 == 0 or i + 1 == len(cases):
                    print(f"{variant} / {policy}: {i + 1}/{len(cases)}", flush=True)
            groups.append(dict(variant=variant, policy=policy, rows=rows, fingerprint=fingerprint,
                               operation_cycles=stages,
                               memory_accesses=memory, memory_sample=cases[0]["name"],
                               clock_mhz=engine.clock_mhz, weight_bytes=weight_bytes,
                               parameter_bytes=parameter_bytes, program_bytes=program_bytes,
                               upload_ms=engine.model_load_ms,
                               per_image_weight_bytes=weight_bytes if policy == "reload" else 0))
            temporary = output.with_suffix(".tmp")
            temporary.write_text(json.dumps(groups, indent=2) + "\n")
            temporary.replace(output)
    return groups


def latency_stats(values):
    values = sorted(values)
    return dict(n=len(values), mean=statistics.mean(values), median=statistics.median(values),
                p95=values[math.ceil(0.95 * len(values)) - 1], minimum=values[0], maximum=values[-1])


def accuracy(rows):
    return sum(r["prediction"] == r["label"] for r in rows) / len(rows)


def select_configuration(groups, floor=0.85):
    """Compare measured latency selection with accuracy-first on a held-out suffix."""
    if { (g["variant"], g["policy"]) for g in groups } != {
            (v, p) for v in VARIANTS for p in POLICIES}:
        raise ValueError("Load all six benchmark runs before selecting a configuration")
    size = len(groups[0]["rows"])
    if size < 2 or any(len(g["rows"]) != size for g in groups):
        raise ValueError("Selection requires equal runs and at least two images")
    split = max(1, min(size - 1, size * 7 // 10))
    answers = []
    for policy in POLICIES:
        candidates = [g for g in groups if g["policy"] == policy and
                      accuracy(g["rows"][:split]) >= floor]
        if not candidates:
            print(f"{policy}: no configuration meets {floor:.0%} on {split} calibration images")
            continue
        estimate = lambda g: statistics.median(r["policy_ms"] for r in g["rows"][:split])
        chosen = min(candidates, key=estimate)
        baseline = max(candidates, key=lambda g: accuracy(g["rows"][:split]))
        for method, group in (("Latency-aware", chosen), ("Accuracy-first", baseline)):
            held = group["rows"][split:]
            answers.append(dict(policy=policy, method=method, variant=group["variant"],
                                n=len(held), accuracy=accuracy(held),
                                predicted_ms=estimate(group),
                                measured_ms=statistics.median(r["policy_ms"] for r in held)))
    return answers


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, default=Path("benchmark_results.json"))
    parser.add_argument("--policy", choices=(*POLICIES, "both"), default="both")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    policies = POLICIES if args.policy == "both" else (args.policy,)
    run_suite(load_suite(), args.output, args.limit, policies, args.resume)
