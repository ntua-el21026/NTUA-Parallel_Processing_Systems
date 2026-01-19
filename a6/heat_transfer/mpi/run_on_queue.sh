#!/bin/bash
#PBS -q parlab
#PBS -N benchmark_mpi
#PBS -l nodes=8:ppn=8
#PBS -l walltime=01:30:00
#PBS -o results_benchmark.txt
#PBS -e error_benchmark.txt


module load openmpi/1.8.3

cd $PBS_O_WORKDIR

EXECUTABLES=("jacobi_mpi" "gauss_mpi" "redblack_mpi")
 
SIZES=(2048 4096 6144)


CONFIGS=(
    "1 1 1"
    "2 2 1"
    "4 2 2"
    "8 4 2"
    "16 4 4"
    "32 8 4"
    "64 8 8"
)

echo "=================================================================="
echo "Starting MPI Benchmarks at $(date)"
echo "Config: T=256 iterations (Hardcoded in source or -D defines)"
echo "=================================================================="

for EXEC in "${EXECUTABLES[@]}"; do
    if [ ! -f "./$EXEC" ]; then
        echo "WARNING: Executable ./$EXEC not found. Skipping."
        continue
    fi

    echo "##################################################################"
    echo "Benchmarking Implementation: $EXEC"
    echo "##################################################################"

    for SIZE in "${SIZES[@]}"; do
        echo "  --> Matrix Size: ${SIZE}x${SIZE}"
        
        for CONF in "${CONFIGS[@]}"; do

            read P Px Py <<< "$CONF"
            
            echo "      Processes: $P (Grid: ${Px}x${Py})"

 
            for (( i=1; i<=3; i++ )); do
              
                mpirun -np $P --mca btl tcp,self ./$EXEC $SIZE $SIZE $Px $Py
                
            done
            echo "      ----------------------------------"
        done
        echo "=================================================================="
    done
done

echo "Benchmarks finished at $(date)"
