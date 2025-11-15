# Project Setup & Benchmarking

This project includes a Python virtual environment and benchmarking scripts for comparing MLP vs Naive KAN architectures.

## Virtual Environment

The project uses a local `.venv` for isolated dependencies.

### Activate the venv

**Windows Command Prompt:**
```cmd
.venv\Scripts\activate
```

**Windows PowerShell:**
```powershell
.venv\Scripts\Activate.ps1
```

**WSL/Linux:**
```bash
source .venv/bin/activate
```

## Project Files

- `test_vtune.py` — Core benchmark script (MLP vs KAN with exact parameter matching)
- `run_all.py` — Master runner script (executes test_vtune.py and stores results)
- `test.py` — Original benchmark script with `--perf` option (Linux only)
- `results/` — Output directory for benchmark results (timestamped files)
- `timing_only.py` — Refactored CLI harness for running deterministic loops (MLP, KAN or both)
- `benchmark_mlp.py` — Convenience wrapper that pins `--model mlp` for profiler/perf runs
- `benchmark_kan.py` — Convenience wrapper that pins `--model kan` for profiler/perf runs

## Perf-Based Cache Analysis (Linux)

The new `benchmark_*.py` entrypoints run a tight forward/backward training loop that is stable under Linux `perf` counters. Example commands (the `sudo` prefix depends on your distro configuration):

```bash
sudo perf stat -e cache-references,cache-misses,cycles,instructions \
	/Users/achintya/Achintya/College_Penn/Sem-1/CSE-530/CSE530-Project/.venv/bin/python \
	benchmark_mlp.py --iterations 200 --warmup 50 --batch-size 32

sudo perf stat -e cache-references,cache-misses,cycles,instructions \
	/Users/achintya/Achintya/College_Penn/Sem-1/CSE-530/CSE530-Project/.venv/bin/python \
	benchmark_kan.py --iterations 200 --warmup 50 --batch-size 32
```

To run both models back-to-back without `perf`, use the shared CLI:

```bash
/Users/achintya/Achintya/College_Penn/Sem-1/CSE-530/CSE530-Project/.venv/bin/python timing_only.py --model both
```

Useful switches:

- `--iterations`: number of timed steps (default 25)
- `--warmup`: warmup steps discarded before `perf` sampling (default 5)
- `--no-backward`: forward-only runs (skip gradient computation)
- `--device`: `cpu` or `cuda:0`
- `--kan-width`: width of the intermediate KAN layer

> **Heads-up:** The `kan` package must be installed (`pip install kan`) for KAN benchmarks. The MLP path works without it.

## Running Benchmarks

### Quick Start: Run and Store Results

The main way to benchmark:

```cmd
.venv\Scripts\python.exe run_all.py         # Quick run (50 steps per section)
.venv\Scripts\python.exe run_all.py --long  # Longer run (200 steps per section)
```

Results are automatically saved to `results/` with:
- `benchmark_YYYYMMDD_HHMMSS.txt` — Full benchmark output
- `benchmark_YYYYMMDD_HHMMSS.json` — Metadata (timestamp, return code, command)

### Direct Benchmark Execution

Run the core benchmark directly:

```cmd
.venv\Scripts\python.exe test_vtune.py         # Quick (50 steps)
.venv\Scripts\python.exe test_vtune.py --long  # Longer (200 steps)
```

## Profiling with Intel VTune

For cache analysis and memory access profiling, use Intel VTune:

```cmd
# Install VTune: https://www.intel.com/content/www.intel.com/en/develop/tools/oneapi/all-toolkits.html

# Profile memory access patterns (cache analysis)
vtune -collect memory-access -app-working-dir . -- .venv\Scripts\python.exe test_vtune.py --long

# Profile CPU hotspots
vtune -collect hotspots -app-working-dir . -- .venv\Scripts\python.exe test_vtune.py --long

# General performance analysis
vtune -collect performance -app-working-dir . -- .venv\Scripts\python.exe test_vtune.py --long
```

**VTune will provide:**
- L1/L2/L3 cache hit/miss rates
- Memory bandwidth utilization
- TLB (translation lookaside buffer) misses
- Hotspot functions and lines
- Memory access patterns

## Benchmark Details

### Model Matching

Both models are created with exactly matched parameters:
- **MLP:** 4 parameter tensors (2 layers)
- **KAN:** 2048 parameter tensors (one network per output dimension)
- **Total params:** ~1.58M (matched within 508 params)

### Benchmark Sections

Each run executes 4 benchmark phases:
1. **MLP forward pass** (50 or 200 iterations)
2. **KAN forward pass** (50 or 200 iterations)
3. **MLP forward+backward** (50 or 200 iterations)
4. **KAN forward+backward** (50 or 200 iterations)

## Dependencies

Installed in `.venv`:
- `torch` — PyTorch (CPU version)
- `numpy`, `scipy` — Scientific computing
- `pandas`, `scikit-learn` — Data science
- `matplotlib` — Visualization
- `requests` — HTTP
- `pytest` — Testing
