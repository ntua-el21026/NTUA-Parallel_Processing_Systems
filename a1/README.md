# a1 - Intro and Game of Life (OpenMP)

This folder contains the first lab assignment for the Parallel Processing Systems course. The focus is getting familiar with the lab environment and parallelizing Conway's Game of Life using OpenMP.

## Assignment focus
- Parallelize the Game of Life update loop with OpenMP and verify correctness.
- Measure execution time and scaling across different problem sizes and thread counts.
- Automate compilation/execution on the lab cluster (queue scripts).

The accompanying report (see `../docs/reports/individual/a1/`) documents the OpenMP strategy, timing methodology, and performance results.

## Contents
- `game_of_life.c`, `life_par.c`: serial and OpenMP versions of the Game of Life.
- `Makefile`: build targets for serial/parallel executables.
- `run_on_queue.sh`, `make_on_queue.sh`: scripts for running on the cluster.
- `benchmarks/`: raw timing outputs.
- `diagrams/`: plots generated from benchmarks.
