#!/bin/bash
#PBS -q parlab
#PBS -N make_mpi
#PBS -l nodes=1:ppn=1
#PBS -l walltime=00:05:00
#PBS -o make_output.txt
#PBS -e make_error.txt


module load openmpi/1.8.3


cd $PBS_O_WORKDIR

echo "Starting Compilation..."

# 1. Compile Jacobi
echo "Compiling Jacobi..."
mpicc -O3 mpi_jacobi.c utils.c -o jacobi_mpi -lm

# 2. Compile Gauss-Seidel
echo "Compiling Gauss-Seidel..."
mpicc -O3 mpi_gauss.c utils.c -o gauss_mpi -lm

# 3. Compile Red-Black
echo "Compiling Red-Black..."
mpicc -O3 mpi_redblack.c utils.c -o redblack_mpi -lm

if [[ -f "jacobi_mpi" && -f "gauss_mpi" && -f "redblack_mpi" ]]; then
    echo "SUCCESS: All executables created successfully."
else
    echo "ERROR: Some compilations failed. Check make_error.txt."
fi
