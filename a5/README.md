# a5 - GPU K-means with CUDA

This assignment moves K-means clustering to NVIDIA GPUs using CUDA. The goal is to compare increasingly optimized GPU kernels and quantify the impact of memory access patterns and host-device transfers.

## Assignment focus
- Naive GPU offload (membership on GPU, centroid update on CPU).
- Data layout optimization (transpose/column-major) for memory coalescing.
- Shared-memory centroids to reduce global memory traffic.
- Full GPU version with atomics for centroid updates.

The report in `../docs/reports/individual/a5/` walks through each kernel variant and the performance analysis.

## Contents
- `cuda_kmeans_naive.cu`, `cuda_kmeans_transpose.cu`, `cuda_kmeans_shared.cu`, `cuda_kmeans_all_gpu.cu`: kernel variants.
- `main_gpu.cu`, `main_sec.c`, `seq_kmeans.c`: drivers and reference code.
- `Makefile`, `make_on_queue.sh`, `run_on_queue.sh`: build and queue scripts.
- `validation_results.txt`, `validate_on_queue.sh`: correctness checks.
- `diagrams/`, `Execution_logs/`: plots and run logs.
