#!/bin/bash

# ==========================================
# PBS Directives
# ==========================================
#PBS -N validate_kmeans
#PBS -o validate.out
#PBS -e validate.err
#PBS -l nodes=silver1:ppn=40
#PBS -q serial
#PBS -l walltime=00:15:00

# ==========================================
# Environment Setup
# ==========================================
source /etc/profile
module load cuda
cd $PBS_O_WORKDIR

# ==========================================

# ==========================================
echo "---------------------------------------------------"
echo "Cleaning and Re-compiling with VALIDATE flag..."
echo "---------------------------------------------------"

make clean

make VALIDATE_FLAG=-DVALIDATE


SIZE=256
COORDS=16
CLUSTERS=16
LOOPS=5
BLOCK_SIZE=256

OUTPUT_FILE="validation_results.txt"

echo "---------------------------------------------------" > $OUTPUT_FILE
echo "STARTING VALIDATION TESTS @ $(date)" >> $OUTPUT_FILE
echo "Size: $SIZE, Coords: $COORDS, Clusters: $CLUSTERS" >> $OUTPUT_FILE
echo "---------------------------------------------------" >> $OUTPUT_FILE

PROGS="kmeans_cuda_naive kmeans_cuda_transpose kmeans_cuda_shared kmeans_cuda_all_gpu"

for PROG in $PROGS; do
    echo "" >> $OUTPUT_FILE
    echo ">>> Testing Implementation: $PROG" >> $OUTPUT_FILE
    echo "-------------------------------------" >> $OUTPUT_FILE
    
    if [ -f "./$PROG" ]; then
        
        ./$PROG -s $SIZE -n $COORDS -c $CLUSTERS -l $LOOPS -b $BLOCK_SIZE >> $OUTPUT_FILE 2>&1
        
        echo "Finished $PROG"
    else
        echo "ERROR: Executable ./$PROG not found!" >> $OUTPUT_FILE
    fi
done

echo "---------------------------------------------------" >> $OUTPUT_FILE
echo "VALIDATION COMPLETED" >> $OUTPUT_FILE
echo "---------------------------------------------------"


echo "Cleaning up validated objects..."
make clean

echo "Done. Check the file '$OUTPUT_FILE' for results."
