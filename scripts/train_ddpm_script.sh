#!/bin/bash
#SBATCH -p gpu -c 16 
#SBATCH --gres=gpu:8 

set -x

source ../../anaconda3/etc/profile.d/conda.sh
conda activate el2nm

python train_ddpm_model.py