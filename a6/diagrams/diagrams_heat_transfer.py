#!/usr/bin/env python3
"""
Generate heat transfer (Jacobi MPI) time and speedup plots from benchmark files.

Usage:
    python diagrams_heat_transfer.py
    python diagrams_heat_transfer.py --benchmarks /path/to/results_benchmark.txt --outdir /path/to/output
    python diagrams_heat_transfer.py --convergence /path/to/validate_output.txt
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402


LINE_RE = re.compile(
    r"^Jacobi\s+X\s+(?P<x>\d+)\s+Y\s+(?P<y>\d+)\s+Px\s+(?P<px>\d+)\s+Py\s+(?P<py>\d+)\s+Iter\s+(?P<iter>\d+)\s+"
    r"ComputationTime\s+(?P<comp>[0-9]*\.?[0-9]+)\s+TotalTime\s+(?P<total>[0-9]*\.?[0-9]+)"
)

CONV_SERIAL_RE = re.compile(
    r"^Jacobi\s+X\s+(?P<x>\d+)\s+Y\s+(?P<y>\d+)\s+Iter\s+(?P<iter>\d+)\s+Time\s+(?P<total>[0-9]*\.?[0-9]+)"
)
CONV_MPI_RE = re.compile(
    r"^Jacobi\s+X\s+(?P<x>\d+)\s+Y\s+(?P<y>\d+)\s+Px\s+(?P<px>\d+)\s+Py\s+(?P<py>\d+)\s+Iter\s+(?P<iter>\d+)\s+"
    r"ComputationTime\s+(?P<comp>[0-9]*\.?[0-9]+)\s+TotalTime\s+(?P<total>[0-9]*\.?[0-9]+)\s+ConvergenceTime\s+(?P<conv>[0-9]*\.?[0-9]+)"
)

EXPECTED_SIZES = [2048, 4096, 6144]
EXPECTED_PROCS = [1, 2, 4, 8, 16, 32, 64]
BAR_PLOT_PROCS = [8, 16, 32, 64]
CONV_SIZE = 512
CONV_PROCS = 64

SEQUENTIAL_COLOR = "#4C78A8"
MPI_COLOR = "#F58518"
TOTAL_COLOR = "#54A24B"
COMP_COLOR = "#E45756"
CONV_COLOR = "#B279A2"


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parents[1]
    default_bench = base_dir / "heat_transfer" / "mpi" / "results_benchmark.txt"
    default_conv = base_dir / "heat_transfer" / "validate_output.txt"
    default_out = Path(__file__).resolve().parent / "images" / "heat_transfer"
    parser = argparse.ArgumentParser(
        description="Create Jacobi MPI time/speedup plots from heat transfer benchmarks."
    )
    parser.add_argument(
        "--benchmarks",
        type=Path,
        default=default_bench,
        help="Path to results_benchmark.txt.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=default_out,
        help="Output directory for generated images.",
    )
    parser.add_argument(
        "--convergence",
        type=Path,
        default=default_conv,
        help="Path to validate_output.txt for convergence-check measurements.",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Image DPI.")
    return parser.parse_args()


def parse_benchmarks(path: Path) -> dict[int, dict[int, dict[str, list[float]]]]:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")
    data: dict[int, dict[int, dict[str, list[float]]]] = {}
    for line in path.read_text(errors="ignore").splitlines():
        match = LINE_RE.match(line.strip())
        if not match:
            continue
        x = int(match.group("x"))
        y = int(match.group("y"))
        if x != y:
            raise ValueError(f"Non-square matrix found: {x}x{y}")
        px = int(match.group("px"))
        py = int(match.group("py"))
        procs = px * py
        comp = float(match.group("comp"))
        total = float(match.group("total"))
        data.setdefault(x, {}).setdefault(procs, {"comp": [], "total": []})
        data[x][procs]["comp"].append(comp)
        data[x][procs]["total"].append(total)
    if not data:
        raise ValueError(f"No Jacobi benchmark lines found in {path}")
    return data


def average(values: list[float]) -> float:
    return sum(values) / len(values)


def summarize(
    raw: dict[int, dict[int, dict[str, list[float]]]]
) -> dict[int, dict[int, dict[str, float]]]:
    summary: dict[int, dict[int, dict[str, float]]] = {}
    for size, per_proc in raw.items():
        summary[size] = {}
        for procs, times in per_proc.items():
            summary[size][procs] = {
                "comp": average(times["comp"]),
                "total": average(times["total"]),
            }
    return summary


def validate_data(summary: dict[int, dict[int, dict[str, float]]]) -> None:
    missing_sizes = [s for s in EXPECTED_SIZES if s not in summary]
    if missing_sizes:
        raise ValueError(f"Missing sizes in benchmarks: {missing_sizes}")
    for size in EXPECTED_SIZES:
        missing_procs = [p for p in EXPECTED_PROCS if p not in summary[size]]
        if missing_procs:
            raise ValueError(f"Missing processes for size {size}: {missing_procs}")


def parse_convergence(path: Path) -> dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(f"Convergence file not found: {path}")
    serial_times: list[float] = []
    mpi_times: dict[int, dict[str, list[float]]] = {}
    for line in path.read_text(errors="ignore").splitlines():
        serial_match = CONV_SERIAL_RE.match(line.strip())
        if serial_match:
            x = int(serial_match.group("x"))
            y = int(serial_match.group("y"))
            if x != y:
                continue
            if x == CONV_SIZE:
                serial_times.append(float(serial_match.group("total")))
            continue
        mpi_match = CONV_MPI_RE.match(line.strip())
        if not mpi_match:
            continue
        x = int(mpi_match.group("x"))
        y = int(mpi_match.group("y"))
        if x != y or x != CONV_SIZE:
            continue
        procs = int(mpi_match.group("px")) * int(mpi_match.group("py"))
        mpi_times.setdefault(procs, {"comp": [], "total": [], "conv": []})
        mpi_times[procs]["comp"].append(float(mpi_match.group("comp")))
        mpi_times[procs]["total"].append(float(mpi_match.group("total")))
        mpi_times[procs]["conv"].append(float(mpi_match.group("conv")))

    if not serial_times:
        raise ValueError(f"No Jacobi serial convergence data found for {CONV_SIZE}x{CONV_SIZE}")
    if CONV_PROCS not in mpi_times:
        raise ValueError(
            f"No Jacobi MPI convergence data found for {CONV_SIZE}x{CONV_SIZE} "
            f"with {CONV_PROCS} processes"
        )

    return {
        "serial_total": average(serial_times),
        "mpi_total": average(mpi_times[CONV_PROCS]["total"]),
        "mpi_comp": average(mpi_times[CONV_PROCS]["comp"]),
        "mpi_conv": average(mpi_times[CONV_PROCS]["conv"]),
    }


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


def plot_speedup(
    size: int,
    times: dict[int, dict[str, float]],
    out_path: Path,
    dpi: int,
) -> None:
    procs = EXPECTED_PROCS
    totals = [times[p]["total"] for p in procs]
    seq_time = totals[0]
    speedups = [seq_time / t for t in totals]
    positions = list(range(len(procs)))
    labels = [str(p) for p in procs]
    colors = [SEQUENTIAL_COLOR] + [MPI_COLOR] * (len(procs) - 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(positions, speedups, color=colors, edgecolor="black")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel("MPI processes")
    ax.set_ylabel("Speedup (sequential time / parallel time)")
    ax.set_title(f"Jacobi MPI Speedup (Matrix {size}x{size})")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    add_bar_labels(ax, bars, fmt="{:.2f}")
    legend_patches = [
        Patch(facecolor=SEQUENTIAL_COLOR, edgecolor="black", label="Sequential"),
        Patch(facecolor=MPI_COLOR, edgecolor="black", label="MPI"),
    ]
    ax.legend(handles=legend_patches)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_time_barplots(
    size: int,
    times: dict[int, dict[str, float]],
    out_dir: Path,
    dpi: int,
) -> None:
    max_value = 0.0
    for procs in BAR_PLOT_PROCS:
        max_value = max(
            max_value,
            times[procs]["total"],
            times[procs]["comp"],
        )
    y_max = max_value * 1.1 if max_value > 0 else 1.0

    for procs in BAR_PLOT_PROCS:
        values = [times[procs]["total"], times[procs]["comp"]]
        labels = ["Total time", "Computation time"]
        colors = [TOTAL_COLOR, COMP_COLOR]
        positions = list(range(len(values)))

        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.bar(positions, values, color=colors, edgecolor="black")
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Time type")
        ax.set_ylabel("Time (s)")
        ax.set_title(f"Jacobi MPI Times (Matrix {size}x{size}, {procs} processes)")
        ax.set_ylim(0, y_max)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
        add_bar_labels(ax, bars)
        legend_patches = [
            Patch(facecolor=TOTAL_COLOR, edgecolor="black", label="Total time"),
            Patch(facecolor=COMP_COLOR, edgecolor="black", label="Computation time"),
        ]
        ax.legend(handles=legend_patches)
        fig.tight_layout()
        out_path = out_dir / f"jacobi_times_{size}_p{procs}.png"
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)


def plot_convergence_times(
    conv: dict[str, float],
    out_path: Path,
    dpi: int,
) -> None:
    values = [conv["mpi_total"], conv["mpi_comp"], conv["mpi_conv"]]
    labels = ["Total time", "Computation time", "Convergence time"]
    colors = [TOTAL_COLOR, COMP_COLOR, CONV_COLOR]
    positions = list(range(len(values)))

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(positions, values, color=colors, edgecolor="black")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Time type")
    ax.set_ylabel("Time (s)")
    ax.set_title(
        f"Jacobi MPI Convergence Check (Matrix {CONV_SIZE}x{CONV_SIZE}, {CONV_PROCS} processes)"
    )
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    add_bar_labels(ax, bars)
    legend_patches = [
        Patch(facecolor=TOTAL_COLOR, edgecolor="black", label="Total time"),
        Patch(facecolor=COMP_COLOR, edgecolor="black", label="Computation time"),
        Patch(facecolor=CONV_COLOR, edgecolor="black", label="Convergence time"),
    ]
    ax.legend(handles=legend_patches)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    raw = parse_benchmarks(args.benchmarks)
    summary = summarize(raw)
    validate_data(summary)

    args.outdir.mkdir(parents=True, exist_ok=True)

    for size in EXPECTED_SIZES:
        plot_speedup(
            size,
            summary[size],
            args.outdir / f"jacobi_speedup_{size}.png",
            args.dpi,
        )
        plot_time_barplots(size, summary[size], args.outdir, args.dpi)

    conv = parse_convergence(args.convergence)
    plot_convergence_times(
        conv,
        args.outdir / f"jacobi_convergence_{CONV_SIZE}_p{CONV_PROCS}.png",
        args.dpi,
    )

    print(f"Wrote plots to: {args.outdir}")


if __name__ == "__main__":
    main()
