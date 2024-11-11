#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

#SBATCH --job-name=env_creation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=70
#SBATCH --mem=300GB
#SBATCH --time=00:30:00
#SBATCH --partition=dev_gpu_4

# Exit immediately if a command exits with a non-zero status
set -e

# Start timer
start_time=$(date +%s)

# Get the current date
current_date=$(date +%y%m%d)

# Create environment name with the current date
env_prefix=lingua_$current_date

# Create the conda environment
module load devel/miniconda/23.9.0-py3.9.15
# source $CONDA_ROOT/etc/profile.d/conda.sh

# check whether env already exists
# Check if the directory exists and remove it if it’s not a Conda environment
env_path="$HOME/.conda/envs/$env_prefix"
if [ -d "$env_path" ] && [ ! -f "$env_path/conda-meta/history" ]; then
    echo "Removing non-environment directory $env_path"
    rm -rf "$env_path"
fi


conda create -n $env_prefix python=3.11 -y -c anaconda
conda activate $env_prefix

echo "Currently in env $(which python)"

# Install packages
pip install torch==2.5.0 xformers --index-url https://download.pytorch.org/whl/cu121
pip install ninja
pip install --requirement requirements.txt

# End timer
end_time=$(date +%s)

# Calculate elapsed time in seconds
elapsed_time=$((end_time - start_time))

# Convert elapsed time to minutes
elapsed_minutes=$((elapsed_time / 60))

echo "Environment $env_prefix created and all packages installed successfully in $elapsed_minutes minutes!"


