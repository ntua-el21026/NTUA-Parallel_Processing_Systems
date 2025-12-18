#!/bin/bash

## Job Name
#PBS -N run_conc_ll

## Output and error of PBS (not the runs)
#PBS -o run_conc_ll.pbs_out
#PBS -e run_conc_ll.pbs_err

## Sandman, serial queue, 64 threads available
#PBS -q serial
#PBS -l nodes=sandman:ppn=64

## Maximum walltime (adjust if necessary)
#PBS -l walltime=01:00:00

## Go to the directory where qsub was executed
# CHANGE THIS TO YOUR ACTUAL DIRECTORY
cd $HOME/a2/conc_ll 

# --- Define Core Parameters ---
IMPLEMENTATIONS="serial cgl fgl opt lazy nb"
NTHREADS="1 2 4 8 16 32 64 128"
LIST_SIZES="1024 8192"

# Workloads: (Contains, Add, Remove)
# Format: "C_A_R"
WORKLOADS="100_0_0 80_10_10 20_40_40 0_50_50"

# Directory for results
OUTDIR="results_conc_ll"
mkdir -p "$OUTDIR"

# --- Helper Function to generate MT_CONF for thread pinning ---
# Generates a comma-separated list of logical core IDs.
# Assumes sandman has 64 logical cores (0-63).
# For N > 64, it cycles through the 64 available logical cores (oversubscription).
get_mt_conf() {
    local N=$1
    local CONFIG=""
    local MAX_LOGICAL_CORES=64 

    for i in $(seq 0 $((N - 1))); do
        # Core ID cycles through 0, 1, ..., 63, 0, 1, ...
        local CORE_ID=$((i % MAX_LOGICAL_CORES))
        
        CONFIG="${CONFIG}${CORE_ID}"
        if [ $i -lt $((N - 1)) ]; then
            CONFIG="${CONFIG},"
        fi
    done
    echo "$CONFIG"
}

# --- Main Execution Loop ---
for IMPL in $IMPLEMENTATIONS; do
    EXECUTABLE="./x.$IMPL"

    for S in $LIST_SIZES; do

        for T in $NTHREADS; do
            
            # For the serial implementation, only run T=1 (to establish baseline)
            if [ "$IMPL" == "serial" ] && [ $T -gt 1 ]; then
                continue 
            fi
            
            # --- MT_CONF Setting for Thread Pinning (pthreads) ---
            if [ $T -gt 1 ]; then
                MT_CONF=$(get_mt_conf $T)
                export MT_CONF
            else
                # Unset MT_CONF for single-threaded execution (T=1)
                unset MT_CONF
            fi
            
            echo "Running $IMPL: ListSize=$S, Nthreads=$T, MT_CONF=$MT_CONF"

            for W in $WORKLOADS; do
                # Split the workload string (e.g., 100_0_0) into C, A, R variables
                IFS='_' read -r C A R <<< "$W"
                
                # Input arguments for the executable: <list_size> <contains_pct> <add_pct> <remove_pct>
                ARGS="$S $C $A $R"
                
                # Output files for this run
                OUT="${OUTDIR}/conc_ll_${IMPL}_S${S}_T${T}_W${W}.out"
                ERR="${OUTDIR}/conc_ll_${IMPL}_S${S}_T${T}_W${W}.err"
                
                # Run the program:
                #  - stdout → OUT
                #  - stderr → ERR
                $EXECUTABLE $ARGS >"$OUT" 2>"$ERR"
            done
        done
    done
done

echo "Execution finished. Results are in the $OUTDIR directory."
