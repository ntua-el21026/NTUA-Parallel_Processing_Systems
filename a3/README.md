# a3 - Synchronization Costs with K-means Locks

This assignment studies mutual exclusion overheads in a shared-memory setting by running K-means under different synchronization strategies.

## Assignment focus
- Compare multiple lock types under the same OpenMP K-means workload.
- Evaluate scaling and lock contention across 1-64 threads.
- Automate benchmarks and generate comparison plots.

The report in `../docs/reports/individual/a3/` details the lock implementations and performance comparisons.

## Contents
- `omp_critical_kmeans.c`, `omp_lock_kmeans.c`, `omp_naive_kmeans.c`: OpenMP variants.
- `locks/`: lock implementations (no-sync, pthread mutex/spin, TAS/TTAS, array, CLH).
- `benchmarks/`: per-lock timing outputs.
- `diagrams/`: plots comparing lock strategies.
- `Makefile`, `run_on_queue.sh`, `make_on_queue.sh`: build and queue scripts.
