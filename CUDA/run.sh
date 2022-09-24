#!/usr/bin/env bash
#SBATCH --job-name=Test
#SBATCH --partition=wacc
#SBATCH --ntasks=1 --cpus-per-task=1
#SBATCH --time=00-00:03:00
#SBATCH --gres=gpu:gtx1080:1
#SBATCH --output=test-%j.out

cd $SLURM_SUBMIT_DIR

module load nvidia/cuda

#nvcc task1.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task1
#./task1

#nvcc task2.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task2
#./task2

nvcc task3.cu vscale.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task3
./task3 2e10
