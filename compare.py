"""Compile and compare the same neural network on CPU and CUDA GPU."""

import argparse
from pathlib import Path
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "python"))

from analysis import analyze, run_benchmark
from build_native import build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-only", action="store_true", help="Measure CPU only, without drawing conclusions about GPU.")
    parser.add_argument("--binary", type=Path, help="Use an existing executable instead of compiling.")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "comparison")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8, 64, 128, 256, 1024])
    parser.add_argument("--repetitions", type=int, default=21)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    if args.repetitions < 3 or args.warmup < 1 or any(n < 1 or n > 100000 for n in args.batches):
        parser.error("Use at least 3 repetitions, 1 warmup, and batch sizes between 1 and 100000.")
    try:
        binary = args.binary.resolve() if args.binary else build(cpu_only=args.cpu_only)
        csv_path = run_benchmark(binary, args.output, args.batches, args.repetitions, args.warmup, cpu_only=args.cpu_only, require_gpu=not args.cpu_only)
        analyze(csv_path)
        print((csv_path.parent / "analysis.md").read_text(encoding="utf-8"))
        print("Chart:", csv_path.parent / "comparison_real.png")
    except (RuntimeError, OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Could not complete: {error}", file=sys.stderr)
        print("CUDA requires a compatible NVIDIA GPU, driver, and nvcc; on Windows, use the Visual Studio x64 Native Tools terminal. To benchmark CPU only: python compare.py --cpu-only", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
