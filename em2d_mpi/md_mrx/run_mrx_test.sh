#!/bin/bash
#PBS -q A_dev
#PBS -P NIFS25KIST067
#PBS -l select=1:ncpus=12:mpiprocs=12:ompthreads=1:mem=6gb
#PBS -l walltime=00:5:00

module load intel/2025.1
module load openmpi/5.0.7/gcc11.5.0

cd ${PBS_O_WORKDIR}

export FC=$(which mpif90)

make clean
make

mpirun ./a.out