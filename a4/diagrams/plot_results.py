#!/usr/bin/env python3
import re
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Filename pattern: conc_ll_<impl>_S<size>_T<threads>_W<r>_<a>_<rm>.out
FNAME_RE = re.compile(
    r"conc_ll_(?P<impl>[^_]+)_S(?P<size>\d+)_T(?P<thr>\d+)_W(?P<w1>\d+)_(?P<w2>\d+)_(?P<w3>\d+)\.out$"
)

TP_RE = re.compile(r"Throughput\(Kops/sec\):\s*([0-9.]+)")
NTH_RE = re.compile(r"Nthreads:\s*(\d+)")

def read_one_file(p: Path):
    m = FNAME_RE.match(p.name)
    if not m:
        return None

    impl = m.group("impl")
    size = int(m.group("size"))
    thr_from_name = int(m.group("thr"))
    workload = f'{m.group("w1")}-{m.group("w2")}-{m.group("w3")}'

    text = p.read_text(errors="ignore")
    tp_m = TP_RE.search(text)
    if not tp_m:
        return None
    thrpt = float(tp_m.group(1))  # Kops/sec

    # Optional sanity: if file also prints Nthreads, compare
    nth_m = NTH_RE.search(text)
    thr_reported = int(nth_m.group(1)) if nth_m else thr_from_name
    if thr_reported != thr_from_name:
        # Keep filename threads as ground truth, but store both
        pass

    return {
        "impl": impl,
        "size": size,
        "threads": thr_from_name,
        "workload": workload,
        "throughput_kops": thrpt,
        "file": p.name,
    }

def plot_group(df_g: pd.DataFrame, outdir: Path, title: str, fname: str):
    # Plot throughput vs threads for all implementations
    plt.figure()
    for impl, d in df_g.groupby("impl"):
        d = d.sort_values("threads")
        plt.plot(d["threads"], d["throughput_kops"], marker="o", label=impl)

    plt.xscale("log", base=2)
    plt.xlabel("Threads (log2)")
    plt.ylabel("Throughput (Kops/sec)")
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / fname, dpi=200)
    plt.close()

def plot_speedup(df_g: pd.DataFrame, outdir: Path, title: str, fname: str):
    # Speedup relative to 1-thread of each implementation (if exists)
    plt.figure()
    for impl, d in df_g.groupby("impl"):
        d = d.sort_values("threads")
        base = d.loc[d["threads"] == 1, "throughput_kops"]
        if base.empty:
            continue
        base = float(base.iloc[0])
        sp = d["throughput_kops"] / base
        plt.plot(d["threads"], sp, marker="o", label=impl)

    plt.xscale("log", base=2)
    plt.xlabel("Threads (log2)")
    plt.ylabel("Speedup vs 1 thread (same impl)")
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / fname, dpi=200)
    plt.close()

def main():
    if len(sys.argv) < 2:
        print("Usage: ./plot_results.py results_conc_ll [outdir=plots]")
        sys.exit(1)

    resdir = Path(sys.argv[1])
    outdir = Path(sys.argv[2]) if len(sys.argv) >= 3 else Path("plots")
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in sorted(resdir.glob("*.out")):
        r = read_one_file(p)
        if r:
            rows.append(r)

    if not rows:
        print("No valid .out files found with expected naming/content.")
        sys.exit(2)

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "all_results.csv", index=False)

    # Throughput plots per (size, workload)
    for (size, workload), df_g in df.groupby(["size", "workload"]):
        title = f"Concurrent Sorted List — Size={size}, Workload={workload}"
        fname = f"throughput_S{size}_W{workload.replace('-','_')}.png"
        plot_group(df_g, outdir, title, fname)

        title2 = f"Speedup — Size={size}, Workload={workload}"
        fname2 = f"speedup_S{size}_W{workload.replace('-','_')}.png"
        plot_speedup(df_g, outdir, title2, fname2)

    print(f"OK: wrote {outdir/'all_results.csv'} and plots into {outdir}/")

if __name__ == "__main__":
    main()
