#!/bin/bash

#PBS -N make_kmeans
#PBS -o make_kmeans.out
#PBS -e make_kmeans.err

#PBS -l nodes=silver1:ppn=40
#PBS -l walltime=00:10:00

cd $PBS_O_WORKDIR

make
