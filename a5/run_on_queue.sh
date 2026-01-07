#!/bin/bash

# ==========================================
# PBS Directives
# ==========================================
#PBS -N full_benchmark_kmeans
#PBS -o benchmark.out
#PBS -e benchmark.err
#PBS -l nodes=silver1:ppn=40
#PBS -l walltime=01:30:00

# ==========================================
# Environment Setup
# ==========================================
module load cuda
cd $PBS_O_WORKDIR


mkdir -p Execution_logs


export CUDA_VISIBLE_DEVICES=0

# ==========================================
# Configuration Parameters 
# ==========================================
SIZE=1024
CLUSTERS=64
LOOPS=10


COORDS_SCENARIOS="32 2"


BLOCK_SIZES="32 48 64 128 238 512 1024"

GPU_PROGS="kmeans_cuda_naive kmeans_cuda_transpose kmeans_cuda_shared kmeans_cuda_all_gpu"

# ==========================================
# Benchmark Loops
# ==========================================

echo "=================================================================="
echo "STARTING BENCHMARKS @ $(date)"
echo "Size: $SIZE, Clusters: $CLUSTERS, Loops: $LOOPS"
echo "=================================================================="

for N in $COORDS_SCENARIOS; do
    echo ""
    echo "------------------------------------------------------------------"
    echo ">>> Running Scenario with Coordinates (Features): $N"
    echo "------------------------------------------------------------------"

    echo "Running Sequential (CPU)..."
    if [ -f "./kmeans_seq" ]; then
        ./kmeans_seq -s $SIZE -n $N -c $CLUSTERS -l $LOOPS
    else
        echo "ERROR: kmeans_seq not found!"
    fi

    for BS in $BLOCK_SIZES; do
        echo "  [Block Size: $BS]"
        
        for PROG in $GPU_PROGS; do
            if [ -f "./$PROG" ]; then
                echo "    Running $PROG..."
                ./$PROG -s $SIZE -n $N -c $CLUSTERS -l $LOOPS -b $BS
            else
                echo "    WARNING: Executable ./$PROG not found. Skipping."
            fi
        done
    done
done

echo ""
echo "=================================================================="
echo "BENCHMARKS COMPLETED @ $(date)"
echo "Check Execution_logs/ folder for CSV files."
echo "=================================================================="
