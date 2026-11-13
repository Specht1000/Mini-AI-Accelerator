"""Generate README figures from actual results saved in results/."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "img"


def training():
    # Learning curve recorded by ai.py during training.
    history = json.loads((ROOT / "results" / "ai_metrics.json").read_text(encoding="utf-8"))["history"]
    epochs = [h["epoch"] for h in history]
    fig, left = plt.subplots(figsize=(8, 3.6))
    left.plot(epochs, [h["loss"] for h in history], "o-", color="#c4513d", label="Error (loss)")
    left.set_xlabel("Epoch (passes over the training set)")
    left.set_ylabel("Training loss", color="#c4513d")
    right = left.twinx()
    right.plot(epochs, [100 * h["validation_accuracy"] for h in history], "s-", color="#1d6b3a")
    right.set_ylabel("Validation accuracy (%)", color="#1d6b3a")
    left.set_title("AI learning: loss decreases and accuracy increases")
    left.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "training.png", dpi=150)
    plt.close(fig)


def cpu():
    # Compare the two CPU implementations measured by the native benchmark.
    rows = list(csv.DictReader(open(ROOT / "results" / "native" / "benchmark.csv", encoding="utf-8")))
    data = {}
    for row in rows:
        if row["algorithm"].startswith("cpu_"):
            data.setdefault(row["algorithm"], {})[int(row["batch"])] = float(row["median_ms"])
    batches = sorted(data["cpu_naive"])
    labels = [str(b) for b in batches]
    x = range(len(batches))
    fig, axis = plt.subplots(figsize=(8, 3.8))
    axis.bar([i - 0.2 for i in x], [data["cpu_naive"][b] for b in batches], 0.4, label="Naive CPU (i-j-k)", color="#9cc0f0")
    axis.bar([i + 0.2 for i in x], [data["cpu_optimized"][b] for b in batches], 0.4, label="Optimized CPU (i-k-j)", color="#1f3b73")
    for i, b in zip(x, batches):
        ratio = data["cpu_naive"][b] / data["cpu_optimized"][b]
        axis.text(i, data["cpu_naive"][b], f"{ratio:.0f}×", ha="center", va="bottom", fontsize=9)
    axis.set_yscale("log")
    axis.set_xticks(list(x), labels)
    axis.set_xlabel("Batch size (samples per call)")
    axis.set_ylabel("Median time (ms, log scale)")
    axis.set_title("Same CPU, same result: only loop order changes")
    axis.legend()
    axis.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "cpu_naive_vs_optimized.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    training()
    cpu()
    print("Figures saved to", OUT)
