#!/bin/bash
#PBS -q A_S
#PBS -P NIFS25KIST067
#PBS -l select=1:ncpus=160:mpiprocs=160:ompthreads=1:mem=400gb
#PBS -l walltime=01:00:00

# 環境をクリーンにして OpenMPI をロード
module purge
module load openmpi/5.0.7/gcc11.5.0

cd ${PBS_O_WORKDIR}

# スタックサイズ制限を解除（念のため）
ulimit -s unlimited

export FC=$(which mpif90)

# 確認用ログ
echo "Current FC: $FC"
echo "Current Directory: $(pwd)"

make clean
make

# 実行
mpirun ./a.out