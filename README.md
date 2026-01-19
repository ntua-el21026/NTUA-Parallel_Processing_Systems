# Parallel_Processing_Systems

Coursework and lab assignments from the Parallel Processing Systems course (NTUA ECE). The repository spans shared-memory, GPU, and distributed-memory programming in C/CUDA with OpenMP and MPI, plus benchmarking scripts and reports.

## Repository structure
- `a1/`: OpenMP Game of Life (introductory parallelization and timing).
- `a2/`: OpenMP K-means and Floyd-Warshall (shared-memory optimization).
- `a3/`: K-means with multiple lock implementations to study synchronization costs.
- `a4/`: Concurrent sorted linked list with coarse/fine/optimistic/lazy/lock-free variants.
- `a5/`: CUDA K-means with progressively optimized GPU kernels.
- `a6/`: MPI K-means and 2D heat transfer (Jacobi and related solvers).
- `docs/`: assignment PDFs, lecture slides, and reports.
- `shell_scripts/`: helper scripts for running on the lab infrastructure.

Each assignment directory contains its own `README.md` with the specific task description, key files, and outputs. The detailed methodology and results are captured in `docs/reports/`.
