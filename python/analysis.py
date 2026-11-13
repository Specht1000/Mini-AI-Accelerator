"""Analyze actual C++/CUDA measurements; never estimate hardware timings."""

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import subprocess

import matplotlib

if __name__ == "__main__":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ai import export_native, train_model

ROOT = Path(__file__).resolve().parent.parent


def run_benchmark(binary, output, batches, repetitions=21, warmup=5, cpu_only=False, require_gpu=False):
    """CPU and GPU run on the same machine with identical data and weights."""
    settings = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    if not (ROOT / "results" / "ai_model.npz").exists():
        train_model(tuple(settings["dimensions"]))
    export_native()
    binary = Path(binary).resolve()
    if not binary.exists():
        raise FileNotFoundError("Compile first with python python/build_native.py [--cpu-only].")

    output = Path(output).resolve()
    command = [str(binary), "--data", str(ROOT / "data" / "native"), "--output", str(output), "--batches", ",".join(str(n) for n in batches), "--repetitions", str(repetitions), "--warmup", str(warmup)]
    if cpu_only:
        command.append("--cpu-only")
    if require_gpu:
        command.append("--require-gpu")
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "4")
    subprocess.run(command, cwd=ROOT, env=environment, check=True)
    (output / "host.json").write_text(json.dumps({"platform": platform.platform(), "processor": platform.processor(), "logical_cpus": os.cpu_count(), "omp_num_threads": environment["OMP_NUM_THREADS"], "command": command}, ensure_ascii=False, indent=2), encoding="utf-8")
    return output / "benchmark.csv"


def load_measurements(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("The CSV contains no measurements.")

    seen = set()
    for row in rows:
        row["batch"] = int(row["batch"])
        row["repetitions"] = int(row["repetitions"])
        for key in ("median_ms", "p10_ms", "p90_ms", "max_error", "accuracy"):
            row[key] = float(row[key])
            if not np.isfinite(row[key]):
                raise ValueError("Non-finite metric in CSV.")
        identity = (row["batch"], row["algorithm"], row["metric"])
        if identity in seen or row["median_ms"] <= 0 or row["repetitions"] < 3:
            raise ValueError("Duplicate measurements, invalid timing, or too few repetitions.")
        if not 0 < row["p10_ms"] <= row["median_ms"] <= row["p90_ms"]:
            raise ValueError("Inconsistent percentiles.")
        if row["max_error"] > 1e-4:
            raise ValueError("Numerical validation failed.")
        seen.add(identity)
    return rows


def analyze(csv_path, output=None):
    """Draw conclusions only for algorithms that actually ran."""
    csv_path = Path(csv_path)
    output = csv_path.parent if output is None else Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = load_measurements(csv_path)
    lookup = {(r["batch"], r["algorithm"], r["metric"]): r for r in rows}
    batches = sorted({row["batch"] for row in rows})
    has_gpu = any(row["algorithm"].startswith("gpu_") for row in rows)
    series = [("cpu_naive", "wall", "Sequential CPU", "#75869a"), ("cpu_optimized", "wall", "Optimized CPU", "#245e91"), ("gpu_naive", "transfer_wall", "Naive CUDA + transfers", "#d28722"), ("gpu_tiled", "transfer_wall", "Tiled CUDA + transfers", "#149986")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for algorithm, metric, label, color in series:
        selected = [lookup[(batch, algorithm, metric)] for batch in batches if (batch, algorithm, metric) in lookup]
        if not selected:
            continue
        x = [row["batch"] for row in selected]
        axes[0].loglog(x, [r["median_ms"] for r in selected], "o-", label=label, color=color)
        axes[0].fill_between(x, [r["p10_ms"] for r in selected], [r["p90_ms"] for r in selected], color=color, alpha=0.15)
    axes[0].set(xlabel="Samples per batch", ylabel="Median wall time (ms)", title="FP32 inference: Dense + bias + ReLU + softmax")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.2)
    report = ["# CPU × CUDA — analysis of actual measurements\n", "Timings use warmups and repeated runs. Bands show the 10th and 90th percentiles, not confidence intervals. Allocation and initial weight loading are excluded on both architectures.\n"]

    if has_gpu:
        report += ["The GPU uses resident weights and pinned host memory. Transfer-inclusive timing covers input H2D, the network, output D2H, and synchronization. Compare CPU with GPU wall times; CUDA event times are shown separately.\n", "| Batch | Optimized CPU (ms) | Tiled CUDA + copies (ms) | CPU/CUDA total | Faster in this measurement |", "|---:|---:|---:|---:|---|"]
        winning = []
        for algorithm, metric, label in (("gpu_tiled", "resident_wall", "Tiled: resident data"), ("gpu_tiled", "transfer_wall", "Tiled: input/output transfers"), ("gpu_naive", "transfer_wall", "Naive: input/output transfers")):
            points = [(batch, lookup[(batch, "cpu_optimized", "wall")]["median_ms"] / lookup[(batch, algorithm, metric)]["median_ms"]) for batch in batches if (batch, algorithm, metric) in lookup]
            axes[1].semilogx([p[0] for p in points], [p[1] for p in points], "o-", label=label)
        for batch in batches:
            if (batch, "gpu_tiled", "transfer_wall") not in lookup:
                continue
            cpu = lookup[(batch, "cpu_optimized", "wall")]["median_ms"]
            gpu = lookup[(batch, "gpu_tiled", "transfer_wall")]["median_ms"]
            if gpu < cpu:
                winning.append(batch)
            winner = "CUDA" if gpu < cpu else "CPU"
            report.append(f"| {batch} | {cpu:.6f} | {gpu:.6f} | {cpu/gpu:.3f}× | {winner} |")
        conclusion = (f"First tested batch where tiled CUDA including transfers wins: **{min(winning)}**." if winning else "Tiled CUDA including transfers did not beat optimized CPU at the tested batch sizes.")
        report.append("\n" + conclusion + " This is not the exact crossover or a guarantee for another GPU.\n")
        axes[1].set(ylabel="Speedup: optimized CPU / CUDA", title="Above 1: CUDA is faster")
    else:
        report += ["**GPU was not run. This file contains no CPU × GPU result.** The comparison below is between two C++ CPU implementations.\n", "| Batch | Sequential CPU (ms) | Optimized CPU (ms) | Sequential/optimized speedup |", "|---:|---:|---:|---:|"]
        ratios = []
        for batch in batches:
            naive = lookup[(batch, "cpu_naive", "wall")]["median_ms"]
            optimized = lookup[(batch, "cpu_optimized", "wall")]["median_ms"]
            ratios.append(naive / optimized)
            report.append(f"| {batch} | {naive:.6f} | {optimized:.6f} | {naive/optimized:.3f}× |")
        axes[1].semilogx(batches, ratios, "o-", label="Sequential / optimized")
        axes[1].set(ylabel="Speedup between CPU versions", title="GPU unavailable: CPU measurements only")

    axes[1].axhline(1, color="black", linestyle="--")
    axes[1].set_xlabel("Samples per batch")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "comparison_real.png", dpi=170)
    plt.close(fig)
    report += ["## Interpretation\n", "Optimized CPU reuses contiguous data and uses OpenMP when enabled at compilation. Tiled CUDA reuses operands in shared memory; padding and barriers can hurt small matrices. The program does not predetermine a winner.\n", "Events measure the interval between stream markers, including all kernels and possible submission gaps. They do not measure a single instruction or total host-visible latency.\n", "Differences small relative to variability require more runs. The benchmark measures warmed-up execution, excluding allocation and weight loading. It does not measure context creation or a cold first call.\n", "Run CPU and GPU on the same machine and preserve metadata.txt and host.json. These educational implementations do not represent peak BLAS, cuBLAS, Tensor Core, or framework performance.\n"]
    (output / "analysis.md").write_text("\n".join(report), encoding="utf-8")
    print("Analysis saved to", output / "analysis.md")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--binary", type=Path, default=ROOT / "build" / ("benchmark.exe" if os.name == "nt" else "benchmark"))
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "native")
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8, 64, 128, 256, 1024])
    parser.add_argument("--repetitions", type=int, default=21)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    if args.run:
        path = run_benchmark(args.binary, args.output, args.batches, args.repetitions, args.warmup, args.cpu_only, args.require_gpu)
    else:
        path = args.csv if args.csv is not None else args.output / "benchmark.csv"
    analyze(path, None if args.csv is not None else args.output)
