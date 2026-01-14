#!/bin/bash

## Give the Job a descriptive name
#PBS -N kmeans_mpi_job

## Output and error files
#PBS -o out_kmeans
#PBS -e error_kmeans

## How many machines should we get? 
## We need 64 cores total for the max case. Clones usually have 8 cores per node.
#PBS -l nodes=8:ppn=8

## Walltime limit
#PBS -l walltime=00:20:00

## Load necessary modules (όπως λέει η εκφώνηση)
module load openmpi/1.8.3

## Navigate to the working directory
cd $PBS_O_WORKDIR

## Create the benchmark output directory
mkdir -p benchmarks_kmeans

## Parameters from exercise:
# Size (-s) = 256 (MB)
# Coords (-n) = 16
# Clusters (-c) = 32
# Loops (-l) = 10 (fixed for benchmarking)

SIZE=256
COORDS=16
CLUSTERS=32
LOOPS=10

echo "Starting K-Means Benchmarks..."
echo "Config: Size=$SIZE, Coords=$COORDS, Clusters=$CLUSTERS, Loops=$LOOPS"

## Loop for different number of processes
for p in 1 2 4 8 16 32 64; do
    echo "Running with $p processes..."
    
    # Δημιουργία ονόματος αρχείου εξόδου
    OUT_FILE="benchmarks_kmeans/kmeans_np${p}.txt"
    
    # Εκτέλεση MPI
    # --mca btl tcp,self: Απαραίτητο για τα clones (αποφυγή sm BTL σε network filesystem)
    mpirun -np $p --mca btl tcp,self ./kmeans_mpi \
        -s $SIZE \
        -n $COORDS \
        -c $CLUSTERS \
        -l $LOOPS \
        > $OUT_FILE
        
    echo "Finished $p processes. Output saved to $OUT_FILE"
done

echo "All benchmarks completed."