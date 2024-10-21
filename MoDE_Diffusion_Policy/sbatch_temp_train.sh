#!/bin/bash

#SBATCH -p accelerated
#SBATCH -A hk-project-sustainebot
#SBATCH -J MoDE_CALVIN
#SBATCH -n 4
#SBATCH -c 16
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --output=/home/hk-project-robolear/ft4740/code/beso_calvin/logs/outputs/new/%x_%j.out
#SBATCH --error=/home/hk-project-robolear/ft4740/code/beso_calvin/logs/outputs//new/%x_%j.err

# Set environment variables
export TORCH_USE_CUDA_DSA=1

# Activate conda environment
source /home/hk-project-robolear/ft4740/miniconda3/bin/activate mode_env 

# mkdir -p "$$TMPDIR"
echo "Job temporary directory: $TMPDIR"

DATA_SSD=$(ws_find data-ssd)
DATASET_PATH="$DATA_SSD/task_ABC_D.tgz"


echo "Extracting dataset to $$TMPDIR..."
if ! tar -xvzf "$DATASET_PATH" -C "$TMPDIR"; then
    echo "Failed to extract dataset"
    exit 1
fi

ls "$TMPDIR"

DATASET_PATH=$(find "$TMPDIR" -type d -name "task_ABC_D" -print | head -n 1)
if [ -z "$DATASET_PATH" ]; then
    echo "Error: Could not find extracted dataset directory"
    exit 1
fi

echo "Using dataset path: $DATASET_PATH"

srun python mode/training.py root_data_dir="$DATASET_PATH" 2>&1 | tee -a "$SLURM_JOB_ID.log"



echo "Job completed"