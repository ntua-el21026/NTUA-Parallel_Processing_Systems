# a2 - Shared-Memory Parallelism (OpenMP)

This assignment explores parallelization and optimization on shared-memory systems using OpenMP. It includes two subprojects: K-means clustering and Floyd-Warshall all-pairs shortest paths.

## Assignment focus
- K-means: implement naive and reduction-based OpenMP variants and study thread affinity.
- Floyd-Warshall: explore serial, striped, and tiled approaches to improve locality and parallelism.
- Collect timing data and compare scalability across thread counts.

The report in `../docs/reports/individual/a2/` summarizes the OpenMP design choices, affinity policies, and performance results.

## Contents
- `kmeans/`: OpenMP K-means implementations, build scripts, and benchmarks.
- `FW/`: Floyd-Warshall implementations (`fw.c`, `fw_sr.c`, `fw_sr_p.c`, `fw_tiled.c`), plus queue scripts and benchmarks.
- `docs/`: assignment PDFs for reference.
