#!/bin/bash --login
#$ -cwd
#SBATCH --job-name=grade_test
#SBATCH --out=base_model.out.%J
#SBATCH --err=base_model.err.%J
#SBATCH --gres=gpu:1
#SBATCH -p gpusmall 


export HDF5_USE_FILE_LOCKING="FALSE"


conda activate /impacs/sad64/miniconda3/envs/pytorchproject

python train.py