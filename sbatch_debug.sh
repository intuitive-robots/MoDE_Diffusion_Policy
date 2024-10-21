#!/bin/bash

#SBATCH -p accelerated
#SBATCH -A hk-project-sustainebot
#SBATCH -J MoDE_CALVIN
#SBATCH -n 4
#SBATCH -c 16
#SBATCH -t 2:00:00
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --output=/home/hk-project-robolear/ft4740/code/beso_calvin/logs/outputs/new/%x_%j.out
#SBATCH --error=/home/hk-project-robolear/ft4740/code/beso_calvin/logs/outputs//new/%x_%j.err

# Set environment variables
export TORCH_USE_CUDA_DSA=1

# Activate conda environment
source /home/hk-project-robolear/ft4740/miniconda3/bin/activate mode_env 

# Create a unique directory for this job within $TMPDIR
JOB_TMPDIR="$TMPDIR"
# mkdir -p "$JOB_TMPDIR"
echo "Job temporary directory: $JOB_TMPDIR"

DATA_SSD=$(ws_find data-ssd)
DATASET_PATH="$DATA_SSD/calvin_debug.tgz"
RESULTS_DIR="$DATA_SSD/results-${SLURM_JOB_ID}"

echo "Extracting dataset to $JOB_TMPDIR..."
if ! tar -xvzf "$DATASET_PATH" -C "$JOB_TMPDIR"; then
    echo "Failed to extract dataset"
    exit 1
fi

ls "$JOB_TMPDIR"

echo "Running training script..."
srun python mode/training.py root_data_dir="$JOB_TMPDIR/calvin_debug_dataset"

echo "Job completed"