#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402


BLOCK_SIZES = [32, 48, 64, 128, 238, 512, 1024]
IMPL_ORDER = ["Naive", "Transpose", "Shmem", "All_GPU"]
VARIANT_IMPLS = [
    ["Naive"],
    ["Naive", "Transpose"],
    ["Naive", "Transpose", "Shmem"],
    ["Naive", "Transpose", "Shmem", "All_GPU"],
]
COO2_VARIANTS = [
    ["Naive", "Transpose", "Shmem"],
    ["Naive", "Transpose", "Shmem", "All_GPU"],
]

GPU_COLOR = "#4C78A8"
TRANSFER_COLOR = "#F58518"
CPU_COLOR = "#54A24B"
SEQUENTIAL_HATCH = "..."
IMPL_HATCHES = {
    "Naive": "",
    "Transpose": "///",
    "Shmem": "\\\\",
    "All_GPU": "xx",
}


def read_sequential_avg(path: Path) -> float:
    rows = []
    with path.open() as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            if row[0].strip().lower() == "sequential":
                rows.append(float(row[2]))
    if not rows:
        raise ValueError(f"No sequential data found in {path}")
    return sum(rows) / len(rows)


def read_impl_avgs(path: Path) -> dict[str, dict[int, float]]:
    data: dict[str, dict[int, float]] = {}
    with path.open() as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            impl = row[0].strip()
            block = int(row[1])
            avg_time = float(row[2])
            data.setdefault(impl, {})[block] = avg_time
    return data


def read_time_breakdown(benchmark_path: Path) -> dict[int, dict[str, dict[int, dict[str, float]]]]:
    dataset_re = re.compile(r"numCoords = (\d+)\s+numClusters = (\d+)")
    block_re = re.compile(r"\[Block Size: (\d+)\]")
    run_re = re.compile(r"Running kmeans_cuda_([a-z_]+)")
    cpu_re = re.compile(r"t_cpu_avg = ([0-9.]+) ms")
    gpu_re = re.compile(r"t_gpu_avg = ([0-9.]+) ms")
    transfers_re = re.compile(r"t_transfers_avg = ([0-9.]+) ms")

    impl_map = {
        "naive": "Naive",
        "transpose": "Transpose",
        "shared": "Shmem",
        "all_gpu": "All_GPU",
        "all_gpu_delta_reduction": "All_GPU_Delta_Reduction",
    }

    current_coords: int | None = None
    current_clusters: int | None = None
    current_block: int | None = None
    current_impl: str | None = None
    cpu = gpu = transfers = None

    collected: dict[int, dict[str, dict[int, list[dict[str, float]]]]] = {}

    for line in benchmark_path.read_text().splitlines():
        dataset_match = dataset_re.search(line)
        if dataset_match:
            current_coords = int(dataset_match.group(1))
            current_clusters = int(dataset_match.group(2))
            continue

        block_match = block_re.search(line)
        if block_match:
            current_block = int(block_match.group(1))
            continue

        run_match = run_re.search(line)
        if run_match:
            current_impl = impl_map.get(run_match.group(1))
            cpu = gpu = transfers = None
            continue

        cpu_match = cpu_re.search(line)
        if cpu_match:
            cpu = float(cpu_match.group(1)) / 1000.0

        gpu_match = gpu_re.search(line)
        if gpu_match:
            gpu = float(gpu_match.group(1)) / 1000.0

        transfers_match = transfers_re.search(line)
        if transfers_match:
            transfers = float(transfers_match.group(1)) / 1000.0

        if (
            current_impl
            and cpu is not None
            and gpu is not None
            and transfers is not None
            and current_coords is not None
            and current_clusters == 64
            and current_coords in (2, 32)
            and current_block is not None
        ):
            collected.setdefault(current_coords, {}).setdefault(current_impl, {}).setdefault(current_block, []).append(
                {"cpu": cpu, "gpu": gpu, "transfers": transfers}
            )
            cpu = gpu = transfers = None

    breakdown: dict[int, dict[str, dict[int, dict[str, float]]]] = {}
    for coords, impls in collected.items():
        for impl, blocks in impls.items():
            for block, entries in blocks.items():
                avg_cpu = sum(e["cpu"] for e in entries) / len(entries)
                avg_gpu = sum(e["gpu"] for e in entries) / len(entries)
                avg_transfers = sum(e["transfers"] for e in entries) / len(entries)
                breakdown.setdefault(coords, {}).setdefault(impl, {})[block] = {
                    "cpu": avg_cpu,
                    "gpu": avg_gpu,
                    "transfers": avg_transfers,
                }
    return breakdown


def plot_breakdown_variant(
    coords: int,
    seq_avg: float,
    breakdown: dict[int, dict[str, dict[int, dict[str, float]]]],
    impl_list: list[str],
    out_path: Path,
) -> None:
    labels = ["Sequential"] + [str(size) for size in BLOCK_SIZES]
    x = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(14, 6))

    group_width = 0.7
    bar_width = group_width / max(1, len(impl_list))
    offsets = [
        (idx - (len(impl_list) - 1) / 2) * bar_width for idx in range(len(impl_list))
    ]

    ax.bar(
        x[0],
        seq_avg,
        width=bar_width,
        color=CPU_COLOR,
        edgecolor="black",
        hatch=SEQUENTIAL_HATCH,
    )

    for impl_idx, impl in enumerate(impl_list):
        impl_data = breakdown.get(coords, {}).get(impl, {})
        if not impl_data:
            raise ValueError(f"No breakdown data found for {impl} coords={coords}")

        gpu_vals = [0.0] + [impl_data[size]["gpu"] for size in BLOCK_SIZES]
        transfer_vals = [0.0] + [impl_data[size]["transfers"] for size in BLOCK_SIZES]
        cpu_vals = [0.0] + [impl_data[size]["cpu"] for size in BLOCK_SIZES]

        positions = [pos + offsets[impl_idx] for pos in x]
        hatch = IMPL_HATCHES.get(impl, "")

        ax.bar(
            positions,
            gpu_vals,
            width=bar_width,
            color=GPU_COLOR,
            edgecolor="black",
            hatch=hatch,
        )
        ax.bar(
            positions,
            transfer_vals,
            width=bar_width,
            bottom=gpu_vals,
            color=TRANSFER_COLOR,
            edgecolor="black",
            hatch=hatch,
        )
        ax.bar(
            positions,
            cpu_vals,
            width=bar_width,
            bottom=[g + t for g, t in zip(gpu_vals, transfer_vals)],
            color=CPU_COLOR,
            edgecolor="black",
            hatch=hatch,
        )

    impl_label = " + ".join(impl_list)
    ax.set_title(
        "K-means Time Breakdown (Sz=1024, Coords={}, Clusters=64)\nImplementations: {}".format(
            coords, impl_label
        )
    )
    ax.set_xlabel("Sequential / Block size")
    ax.set_ylabel("Time per loop (s)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    segment_legend = ax.legend(
        handles=[
            Patch(facecolor=GPU_COLOR, label="GPU time"),
            Patch(facecolor=TRANSFER_COLOR, label="Transfer time"),
            Patch(facecolor=CPU_COLOR, label="CPU time"),
        ],
        loc="upper left",
        frameon=True,
        title="Time Components",
    )
    ax.add_artist(segment_legend)

    impl_handles = [
        Patch(
            facecolor=CPU_COLOR,
            edgecolor="black",
            hatch=SEQUENTIAL_HATCH,
            label="Sequential",
        )
    ]
    for impl in impl_list:
        impl_handles.append(
            Patch(
                facecolor=CPU_COLOR,
                edgecolor="black",
                hatch=IMPL_HATCHES.get(impl, ""),
                label=impl,
            )
        )
    ax.legend(
        handles=impl_handles,
        loc="upper right",
        frameon=True,
        title="Implementations",
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_speedup(
    coords: int,
    seq_avg: float,
    impl_avgs: dict[str, dict[int, float]],
    impl_list: list[str],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))

    for impl in impl_list:
        if impl not in impl_avgs:
            continue
        times = [impl_avgs[impl][size] for size in BLOCK_SIZES]
        speedups = [seq_avg / t for t in times]
        ax.plot(BLOCK_SIZES, speedups, marker="o", linewidth=2, label=impl)

    impl_label = " + ".join(impl_list)
    ax.set_title(
        "Speedup vs Block Size (Sz=1024, Coords={}, Clusters=64)\nImplementations: {}".format(
            coords, impl_label
        )
    )
    ax.set_xlabel("Block size")
    ax.set_ylabel("Speedup (seq_time / parallel_time)")
    ax.set_xticks(BLOCK_SIZES)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent
    logs_dir = base_dir / "Execution_logs"
    images_dir = Path(__file__).resolve().parent / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    seq_32_path = logs_dir / "Sz-1024_Coo-32_Cl-64.csv"
    seq_2_path = logs_dir / "Sz-1024_Coo-2_Cl-64.csv"
    gpu_32_path = logs_dir / "silver1-V100_Sz-1024_Coo-32_Cl-64.csv"
    gpu_2_path = logs_dir / "silver1-V100_Sz-1024_Coo-2_Cl-64.csv"
    benchmark_path = base_dir / "benchmark.out"

    for path in [seq_32_path, seq_2_path, gpu_32_path, gpu_2_path, benchmark_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    seq_32_avg = read_sequential_avg(seq_32_path)
    seq_2_avg = read_sequential_avg(seq_2_path)

    gpu_32_avgs = read_impl_avgs(gpu_32_path)
    gpu_2_avgs = read_impl_avgs(gpu_2_path)

    breakdown = read_time_breakdown(benchmark_path)

    for idx, impl_list in enumerate(VARIANT_IMPLS, start=1):
        variant_tag = "_".join(impl.lower() for impl in impl_list)

        plot_breakdown_variant(
            coords=32,
            seq_avg=seq_32_avg,
            breakdown=breakdown,
            impl_list=impl_list,
            out_path=images_dir / f"time_breakdown_coo32_{variant_tag}.png",
        )

        plot_speedup(
            coords=32,
            seq_avg=seq_32_avg,
            impl_avgs=gpu_32_avgs,
            impl_list=impl_list,
            out_path=images_dir / f"speedup_coo32_{variant_tag}.png",
        )

    for impl_list in COO2_VARIANTS:
        variant_tag = "_".join(impl.lower() for impl in impl_list)

        plot_speedup(
            coords=2,
            seq_avg=seq_2_avg,
            impl_avgs=gpu_2_avgs,
            impl_list=impl_list,
            out_path=images_dir / f"speedup_coo2_{variant_tag}.png",
        )

        plot_breakdown_variant(
            coords=2,
            seq_avg=seq_2_avg,
            breakdown=breakdown,
            impl_list=impl_list,
            out_path=images_dir / f"time_breakdown_coo2_{variant_tag}.png",
        )


if __name__ == "__main__":
    main()
