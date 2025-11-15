#!/usr/bin/env python3
"""Minimal benchmarking harness for KAN vs MLP."""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    """Tiny MLP used for apples-to-apples comparison with KAN."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


@dataclass
class BenchmarkConfig:
    input_dim: int = 64
    output_dim: int = 64
    hidden_dim: int = 128
    kan_width: int = 6
    batch_size: int = 16
    warmup_steps: int = 5
    timed_steps: int = 25
    device: str = "cpu"
    backward: bool = True
    seed: int | None = 42


@dataclass
class BenchmarkResult:
    model_kind: str
    avg_ms: float
    std_ms: float
    iterations: int
    param_count: int
    times_ms: List[float]


def _resolve_device(device: str) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no GPU detected. Use --device cpu.")
    return resolved


def _synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _maybe_import_kan():
    try:
        from kan import KAN  # type: ignore
    except ImportError as exc:  # pragma: no cover - guard for missing optional dep
        raise RuntimeError(
            "KAN package is not installed. Install it via `pip install kan` to run KAN benchmarks."
        ) from exc
    return KAN


def _build_model(model_kind: str, config: BenchmarkConfig) -> nn.Module:
    if model_kind == "mlp":
        return SimpleMLP(config.input_dim, config.hidden_dim, config.output_dim)
    if model_kind == "kan":
        KAN = _maybe_import_kan()
        return KAN([config.input_dim, config.kan_width, config.output_dim])
    raise ValueError(f"Unsupported model kind: {model_kind}")


def _run_steps(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    criterion: nn.Module,
    steps: int,
    backward: bool,
    device: torch.device,
) -> List[float]:
    model.train(True)
    times_ms: List[float] = []
    for _ in range(steps):
        model.zero_grad(set_to_none=True)
        start = time.perf_counter()
        out = model(x)
        loss = criterion(out, y)
        if backward:
            loss.backward()
        _synchronize_if_needed(device)
        end = time.perf_counter()
        times_ms.append((end - start) * 1_000)
    return times_ms


def run_single_benchmark(model_kind: str, config: BenchmarkConfig) -> BenchmarkResult:
    device = _resolve_device(config.device)
    if config.seed is not None:
        torch.manual_seed(config.seed)
    model = _build_model(model_kind, config).to(device)
    params = sum(p.numel() for p in model.parameters())
    x = torch.randn(config.batch_size, config.input_dim, device=device)
    y = torch.randn(config.batch_size, config.output_dim, device=device)
    criterion = nn.MSELoss()

    # Warmup runs keep perf counters stable prior to measurement.
    _run_steps(model, x, y, criterion, config.warmup_steps, config.backward, device)
    times = _run_steps(model, x, y, criterion, config.timed_steps, config.backward, device)

    avg_ms = sum(times) / len(times)
    std_ms = statistics.pstdev(times) if len(times) > 1 else 0.0
    return BenchmarkResult(
        model_kind=model_kind,
        avg_ms=avg_ms,
        std_ms=std_ms,
        iterations=config.timed_steps,
        param_count=params,
        times_ms=times,
    )


def _format_result(result: BenchmarkResult) -> str:
    spread = f"± {result.std_ms:.3f}" if result.std_ms else ""
    return (
        f"{result.model_kind.upper()} — {result.avg_ms:.3f} ms {spread} "
        f"over {result.iterations} iterations ({result.param_count:,} params)"
    )


def _print_ratio(results: Sequence[BenchmarkResult]) -> None:
    lookup = {r.model_kind: r for r in results}
    if "mlp" in lookup and "kan" in lookup and lookup["mlp"].avg_ms > 0:
        ratio = lookup["kan"].avg_ms / lookup["mlp"].avg_ms
        print(f"KAN / MLP ratio: {ratio:.2f}x (KAN is {ratio:.2f}x slower)")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark KAN vs MLP with consistent, perf-friendly loops.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", choices=["mlp", "kan", "both"], default="both")
    parser.add_argument("--input-dim", type=int, default=64)
    parser.add_argument("--output-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--kan-width", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-seed", action="store_true", help="Disable deterministic seeding")
    parser.add_argument("--no-backward", action="store_true", help="Skip backward pass")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> Sequence[BenchmarkResult]:
    args = parse_args(argv)
    seed = None if args.no_seed else args.seed
    config = BenchmarkConfig(
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        hidden_dim=args.hidden_dim,
        kan_width=args.kan_width,
        batch_size=args.batch_size,
        warmup_steps=args.warmup,
        timed_steps=args.iterations,
        device=args.device,
        backward=not args.no_backward,
        seed=seed,
    )
    targets: Iterable[str]
    if args.model == "both":
        targets = ("mlp", "kan")
    else:
        targets = (args.model,)

    results = []
    print("Starting benchmark...\n")
    for kind in targets:
        print(f"Running {kind.upper()}...", flush=True)
        result = run_single_benchmark(kind, config)
        print(_format_result(result))
        results.append(result)
        print("")

    _print_ratio(results)
    return results


if __name__ == "__main__":
    main()
