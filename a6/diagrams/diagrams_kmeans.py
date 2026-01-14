#!/usr/bin/env python3
"""
Generate K-means MPI time and speedup barplots from benchmark files.

Usage:
    python diagrams_kmeans.py
    python diagrams_kmeans.py --benchmarks /path/to/benchmarks_kmeans --outdir /path/to/output
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402


FILE_RE = re.compile(r"kmeans_np(\d+)\.txt$")
TOTAL_RE = re.compile(r"total\s*=\s*([0-9]*\.?[0-9]+)s")
DATASET_RE = re.compile(
    r"dataset_size\s*=\s*([0-9.]+)\s*MB\s+numObjs\s*=\s*(\d+)\s+numCoords\s*=\s*(\d+)\s+numClusters\s*=\s*(\d+)"
)
NLOOPS_RE = re.compile(r"nloops\s*=\s*(\d+)")

EXPECTED_PROCS = [1, 2, 4, 8, 16, 32, 64]
SEQUENTIAL_COLOR = "#4C78A8"
MPI_COLOR = "#F58518"


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parents[1]
    default_bench = base_dir / "kmeans" / "benchmarks_kmeans"
    default_out = Path(__file__).resolve().parent / "images" / "kmeans"
    parser = argparse.ArgumentParser(
        description="Create K-means MPI time/speedup barplots from benchmark files."
    )
    parser.add_argument(
        "--benchmarks",
        type=Path,
        default=default_bench,
        help="Path to benchmarks_kmeans directory.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=default_out,
        help="Output directory for generated images.",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Image DPI.")
    return parser.parse_args()


def read_total_time(path: Path) -> float:
    text = path.read_text(errors="ignore")
    match = TOTAL_RE.search(text)
    if not match:
        raise ValueError(f"Could not find total time in {path}")
    return float(match.group(1))


def read_config(path: Path) -> dict[str, str]:
    text = path.read_text(errors="ignore")
    dataset_match = DATASET_RE.search(text)
    nloops_match = NLOOPS_RE.search(text)
    if not dataset_match or not nloops_match:
        return {}
    size_mb, num_objs, num_coords, num_clusters = dataset_match.groups()
    return {
        "size_mb": size_mb,
        "num_objs": num_objs,
        "num_coords": num_coords,
        "num_clusters": num_clusters,
        "nloops": nloops_match.group(1),
    }


def collect_times(bench_dir: Path) -> dict[int, float]:
    if not bench_dir.exists():
        raise FileNotFoundError(f"Benchmarks directory not found: {bench_dir}")
    times: dict[int, float] = {}
    for path in sorted(bench_dir.glob("kmeans_np*.txt")):
        match = FILE_RE.match(path.name)
        if not match:
            continue
        procs = int(match.group(1))
        times[procs] = read_total_time(path)
    if not times:
        raise ValueError(f"No benchmark files found in {bench_dir}")
    missing = [p for p in EXPECTED_PROCS if p not in times]
    if missing:
        raise ValueError(f"Missing benchmarks for processes: {missing}")
    return times


def format_config(config: dict[str, str]) -> str:
    if not config:
        return "Config: (unknown)"
    return (
        "Config: "
        f"Size={config['size_mb']} MB, "
        f"Objs={config['num_objs']}, "
        f"Coords={config['num_coords']}, "
        f"Clusters={config['num_clusters']}, "
        f"Loops={config['nloops']}"
    )


def add_bar_labels(ax: plt.Axes, bars, fmt: str = "{:.3f}") -> None:
    for rect in bars:
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            height,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_barplot(
    values: list[float],
    labels: list[str],
    title: str,
    ylabel: str,
    out_path: Path,
    dpi: int,
) -> None:
    positions = list(range(len(values)))
    colors = [SEQUENTIAL_COLOR] + [MPI_COLOR] * (len(values) - 1)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(positions, values, color=colors, edgecolor="black")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Sequential / MPI processes")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    add_bar_labels(ax, bars)
    legend_patches = [
        Patch(facecolor=SEQUENTIAL_COLOR, edgecolor="black", label="Sequential"),
        Patch(facecolor=MPI_COLOR, edgecolor="black", label="MPI"),
    ]
    ax.legend(handles=legend_patches)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    times = collect_times(args.benchmarks)
    config = read_config(args.benchmarks / "kmeans_np1.txt")

    procs = EXPECTED_PROCS
    labels = ["Sequential"] + [str(p) for p in procs[1:]]
    values = [times[p] for p in procs]

    seq_time = times[1]
    speedups = [seq_time / t for t in values]

    args.outdir.mkdir(parents=True, exist_ok=True)

    config_line = format_config(config)
    time_title = f"K-means MPI Execution Time\n{config_line}"
    speedup_title = f"K-means MPI Speedup\n{config_line}"

    plot_barplot(
        values,
        labels,
        time_title,
        "Time (s)",
        args.outdir / "kmeans_time.png",
        args.dpi,
    )
    plot_barplot(
        speedups,
        labels,
        speedup_title,
        "Speedup (sequential time / parallel time)",
        args.outdir / "kmeans_speedup.png",
        args.dpi,
    )

    print(f"Wrote: {args.outdir / 'kmeans_time.png'}")
    print(f"Wrote: {args.outdir / 'kmeans_speedup.png'}")


if __name__ == "__main__":
    main()
