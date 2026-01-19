# a6 - Distributed Memory (MPI)

This assignment targets distributed-memory parallelism using MPI. It includes two workloads: K-means clustering and 2D heat transfer.

## Assignment focus
- MPI K-means: distribute points, compute local contributions, and combine centroids with collective reductions.
- 2D heat transfer: grid decomposition with halo exchanges and convergence checks (Jacobi, with Gauss-Seidel and Red-Black variants available in the codebase).
- Measure total time vs computation time and study speedup across process counts.

The report in `../docs/reports/individual/a6/` summarizes the MPI design, timing methodology, and scalability results.

## Contents
- `kmeans/`: MPI K-means implementation, benchmarks, and run scripts.
- `heat_transfer/`: MPI heat transfer kernels and benchmarks (Jacobi, Gauss-Seidel SOR, Red-Black SOR).
- `diagrams/`: plotting scripts and generated figures.
- `docs/`: assignment PDFs.
