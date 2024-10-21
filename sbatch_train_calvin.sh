#!/bin/bash

#SBATCH -p accelerated
#SBATCH -A hk-project-sustainebot
#SBATCH -J MoDE_Baseline

# Cluster Settings
#SBATCH -n 4       # Number of tasks
#SBATCH -c 16  # Number of cores per task
#SBATCH -t 10:00:00 ## 1-00:30:00 # 06:00:00 # 1-00:30:00 # 2-00:00:00
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4


# Define the paths for storing output and error files
#SBATCH --output=/home/hk-project-robolear/ft4740/code/beso_calvin/logs/outputs/%x_%j.out
#SBATCH --error=/home/hk-project-robolear/ft4740/code/beso_calvin/logs/outputs/%x_%j.err


# -------------------------------

# Activate the virtualenv / conda environment
conda activate mode_env

export TORCH_USE_CUDA_DSA=1
# NNODES=1
# NODE_RANK=0
# PORT=29500
# MASTER_ADDR=127.0.0.1
#CUDA_VISIBLE_DEVICES=0,1,2,3  

srun python mode/training.py 