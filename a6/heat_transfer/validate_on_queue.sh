#!/bin/bash
#PBS -q parlab
#PBS -N validate_global
#PBS -l nodes=8:ppn=8
#PBS -l walltime=00:10:00
#PBS -o validate_output.txt
#PBS -e validate_error.txt

# Φόρτωση Περιβάλλοντος
module purge
module load openmpi/1.8.3

# Μετάβαση στον γονικό φάκελο (Parent Directory)
cd $PBS_O_WORKDIR

# Παράμετροι Validation
SIZE=512
TOTAL_PROCS=64
GRID_X=8
GRID_Y=8

echo "=========================================================="
echo "      GLOBAL VALIDATION SCRIPT (Parent Directory)"
echo "      Serial Path: ./serial/"
echo "      MPI Path:    ./mpi/"
echo "      Flag:        -DTEST_CONV"
echo "=========================================================="

# ---------------------------------------------------------
# ΒΗΜΑ 1: Compilation (Σειριακά & MPI)
# ---------------------------------------------------------

echo "Starting Compilation..."

# --- JACOBI ---
echo "[1/3] Compiling Jacobi..."
# Serial
gcc -O3 serial/Jacobi_serial.c serial/utils.c -o serial/jacobi_serial -lm -DTEST_CONV
# MPI
mpicc -O3 mpi/mpi_jacobi.c mpi/utils.c -o mpi/jacobi_mpi -lm -DTEST_CONV

echo "----------------------------------------------------------"

# ---------------------------------------------------------
# ΒΗΜΑ 2: Εκτέλεση Validation
# ---------------------------------------------------------

# Ορίζουμε τα paths των εκτελέσιμων
SERIAL_PATHS=("serial/jacobi_serial")
MPI_PATHS=("mpi/jacobi_mpi")
NAMES=("Jacobi")

for i in 0; do
    S_PATH=${SERIAL_PATHS[$i]}
    M_PATH=${MPI_PATHS[$i]}
    NAME=${NAMES[$i]}

    echo ""
    echo "##########################################################"
    echo "VALIDATING: $NAME"
    echo "##########################################################"

    # Έλεγχος Serial
    if [ -f "./$S_PATH" ]; then
        echo ">>> Reference (Serial Code - 1 Process):"
        ./$S_PATH $SIZE $SIZE
    else
        echo "ERROR: Serial executable ./$S_PATH not found (Compilation failed?)"
    fi

    echo "      ---------------- vs ----------------"

    # Έλεγχος MPI
    if [ -f "./$M_PATH" ]; then
        echo ">>> Target (MPI Code - 64 Processes):"
        mpirun -np $TOTAL_PROCS --mca btl tcp,self ./$M_PATH $SIZE $SIZE $GRID_X $GRID_Y
    else
        echo "ERROR: MPI executable ./$M_PATH not found (Compilation failed?)"
    fi
    
    echo "##########################################################"
done

echo ""
echo "All validations finished "
