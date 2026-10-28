"""Warmup-aware benchmark for the bounded LPP image-analysis pipeline."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from mcrd_accel.config import LPPConfig
from mcrd_accel.lpp import analyze, compute_window_grid


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=root / "reports/benchmark_lpp.json")
    args = parser.parse_args()
    if min(args.size, args.window, args.stride, args.batch_size, args.repeats) <= 0:
        parser.error("sizes, batch size, and repeats must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")

    y, x = np.mgrid[: args.size, : args.size]
    theta = np.deg2rad(32.0)
    image = np.cos(2 * np.pi * (x * np.cos(theta) + y * np.sin(theta)) / 16.0)
    config = LPPConfig(
        w=args.window,
        dL=args.stride,
        minL=4.0,
        maxL=min(32.0, args.window / 2),
        batch_size=args.batch_size,
    )
    grid = compute_window_grid(image.shape, config.w, config.dL)
    for _ in range(args.warmup):
        analyze(image, config)

    durations: list[float] = []
    peak_memory = 0
    for _ in range(args.repeats):
        tracemalloc.start()
        tracemalloc.reset_peak()
        start = time.perf_counter()
        result = analyze(image, config)
        durations.append(time.perf_counter() - start)
        peak_memory = max(peak_memory, tracemalloc.get_traced_memory()[1])
        tracemalloc.stop()

    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    record = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": revision,
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "device": "CPU",
        },
        "configuration": {
            "image_shape": list(image.shape),
            "window": config.w,
            "stride": config.dL,
            "batch_size": config.batch_size,
            "dtype": str(image.dtype),
            "backend": "numpy",
        },
        "warmup_count": args.warmup,
        "repeat_count": args.repeats,
        "setup_seconds": 0.0,
        "steady_state_seconds": float(np.mean(durations)),
        "steady_state_std_seconds": float(np.std(durations)),
        "window_count": grid.n_windows,
        "windows_per_second": grid.n_windows / float(np.mean(durations)),
        "peak_memory_bytes": int(peak_memory),
        "peak_memory_method": "tracemalloc_host_allocations",
        "valid_window_count": int(np.count_nonzero(result.validity)),
        "status": "completed",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
