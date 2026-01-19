# a4 - Concurrent Sorted Linked List

This assignment focuses on concurrent data structures. The target workload is a sorted singly linked list with `contains`, `add`, and `remove` operations under multiple synchronization schemes.

## Assignment focus
- Implement and compare coarse-grain, fine-grain, optimistic, lazy, and lock-free variants.
- Measure throughput under different workloads and thread counts.
- Analyze contention and scalability trade-offs.

The report in `../docs/reports/individual/a4/` documents the synchronization strategies and benchmark results.

## Contents
- `conc_ll/`: implementations and build/run scripts for each synchronization method.
- `benchmarks/`: throughput measurements.
- `diagrams/`: performance plots.
- `docs/`: assignment PDFs.
