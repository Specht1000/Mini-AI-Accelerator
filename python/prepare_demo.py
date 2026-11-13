"""Prepare notebooks and a portable ZIP to benchmark a real GPU."""

from pathlib import Path
import zipfile

import nbformat

from ai import export_native


ROOT = Path(__file__).resolve().parent.parent


def notebook(path, cells):
    content = [nbformat.v4.new_markdown_cell(source) if kind == "md" else nbformat.v4.new_code_cell(source) for kind, source in cells]
    document = nbformat.v4.new_notebook(cells=content)
    document.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nbformat.write(document, path)


def prepare():
    export_native()
    folder = ROOT / "notebooks"
    folder.mkdir(exist_ok=True)
    setup = '''from pathlib import Path
import sys

# Locate the project from its root or notebooks directory.
ROOT = Path.cwd()
if not (ROOT / "python" / "ai.py").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "python"))

from IPython.display import Image, display, Markdown
from ai import MLP
from analysis import run_benchmark, analyze
%matplotlib inline
'''
    notebook(folder / "CPU_GPU.ipynb", [
        ("md", "# C++ and CUDA: actual measurements\n\nThe AI is trained in Python. Inference runs in `native/cpu.cpp` and `native/gpu.cu`. Use CPU-only mode on machines without a compatible NVIDIA GPU. Use `CUDA_Colab.ipynb` to compare CPU and CUDA on the same remote machine. Timings come from hardware execution."),
        ("code", setup),
        ("md", "## 1. Compile CPU code\nThe helper uses an installed C++ compiler. Zig is also supported for CPU-only builds. Native inference is separate from training and analysis."),
        ("code", '''from build_native import build

# Compilation is excluded from the measured timings.
binary = build(cpu_only=True)
'''),
        ("md", "## 2. Full inference and numerical parity\nUse identical FP32 weights and inputs exported from Python. The executable tests rectangular matrix products and checks final probabilities against the reference. It then performs five warmups and 21 repetitions per batch."),
        ("code", '''path = run_benchmark(binary, ROOT / "results" / "native", batches=[1, 8, 64, 128, 256, 1024], cpu_only=True)
rows = analyze(path)
display(Image(filename=str(path.parent / "comparison_real.png")))
display(Markdown((path.parent / "analysis.md").read_text(encoding="utf-8")))
'''),
        ("md", "## 3. What to compare when a GPU is available\n\n- `device_events`: interval between CUDA stream events for the whole network.\n- `resident_wall`: wall time for resident-data inference and synchronization.\n- `transfer_wall`: H2D + network + D2H + synchronization, with resident weights.\n\nCompare CPU `wall` with GPU `transfer_wall` for input/output-inclusive cost. Use the same machine. Speedup above 1 means the GPU was faster in this measurement; no winner is predetermined."),
    ])

    notebook(folder / "CUDA_Colab.ipynb", [
        ("md", "# C++ CPU × CUDA GPU — real hardware\n\n1. Open this notebook in Colab.\n2. Select an **NVIDIA GPU** runtime, if available.\n3. Run the cells in order.\n4. Upload `dist/cpu_gpu_cuda.zip` when prompted.\n\nGPU availability varies. The CPU being measured belongs to the same remote machine as the GPU. The notebook contains no fabricated CUDA results."),
        ("code", '''import subprocess

# Stop if the runtime cannot see an NVIDIA driver/GPU.
subprocess.run(["nvidia-smi"], check=True)
subprocess.run(["nvcc", "--version"], check=True)
'''),
        ("code", '''from google.colab import files
from pathlib import Path
import io
import zipfile
import sys

# Upload only the package created by python/prepare_demo.py.
uploaded = files.upload()
payload = uploaded["cpu_gpu_cuda.zip"]
base = Path("/content").resolve()

with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    # Reject paths that escape the extraction directory.
    for member in archive.infolist():
        target = (base / member.filename).resolve()
        if not target.is_relative_to(base):
            raise ValueError("Invalid path in ZIP archive.")
    archive.extractall(base)

ROOT = base / "cpu_gpu_cuda"
sys.path.insert(0, str(ROOT / "python"))
'''),
        ("md", "## Compile C++ and CUDA together\n`native/gpu.cu` contains naive and tiled matrix products with shared buffers and barriers. The optimized CPU uses OpenMP in this Linux build. The executable fails if numerical validation fails."),
        ("code", '''import os
from build_native import build

os.environ["OMP_NUM_THREADS"] = "4"
binary = build(cpu_only=False)
subprocess.run([str(binary), "--self-test", "--require-gpu"], check=True)
'''),
        ("md", "## Measure the same model on both architectures\nWeights and buffers are prepared before timing. This run requires an actual GPU. It uses five warmups and 21 repetitions per batch."),
        ("code", '''from analysis import run_benchmark, analyze
from IPython.display import Image, display, Markdown
%matplotlib inline

path = run_benchmark(binary, ROOT / "results" / "native", batches=[1, 8, 64, 128, 256, 1024], repetitions=21, warmup=5, require_gpu=True)
rows = analyze(path)
display(Image(filename=str(path.parent / "comparison_real.png")))
display(Markdown((path.parent / "analysis.md").read_text(encoding="utf-8")))
'''),
        ("md", "## Download results\nThe ZIP includes CSV, report, chart, GPU description, and host metadata. Keep percentiles and configuration when presenting results. Tiled CUDA can lose to the naive kernel on small matrices; the code does not force a winner."),
        ("code", '''import shutil

archive = shutil.make_archive("/content/cuda_results", "zip", path.parent)
files.download(archive)
'''),
    ])

    destination = ROOT / "dist"
    destination.mkdir(exist_ok=True)
    selected = ["native/native.hpp", "native/cpu.cpp", "native/gpu.cu", "native/benchmark.cpp", "CMakeLists.txt", "compare.py", "python/build_native.py", "python/ai.py", "python/analysis.py", "config.json", "requirements.txt", "README.md", "CUDA_ZERO.md", "CODE_FLOW.md", "RUN.md", "CODE.md", "CUDA_SETUP.md", "results/ai_model.npz", "results/ai_test.npz", "results/ai_metrics.json"]
    selected += [str(path.relative_to(ROOT)) for path in (ROOT / "data" / "native").glob("*") if path.is_file()]
    selected += [str(path.relative_to(ROOT)) for path in (ROOT / "docs" / "img").glob("*") if path.is_file() and path.suffix in {".svg", ".png"}]

    # Explicit packaging excludes caches, executables, and personal files.
    with zipfile.ZipFile(destination / "cpu_gpu_cuda.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in selected:
            archive.write(ROOT / name, "cpu_gpu_cuda/" + name.replace("\\", "/"))
    print("Notebooks and CUDA package ready in", destination)


if __name__ == "__main__":
    prepare()
