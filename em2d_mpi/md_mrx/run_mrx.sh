#!/bin/bash
#PBS -q A_S
#PBS -P NIFS25KIST067
#PBS -l select=1:ncpus=160:mpiprocs=160:ompthreads=1:mem=400gb
#PBS -l walltime=01:00:00

module load intel/2025.1
module load openmpi/5.0.7/gcc11.5.0

cd ${PBS_O_WORKDIR}

export FC=$(which mpif90)

make clean
make

mpirun ./a.out