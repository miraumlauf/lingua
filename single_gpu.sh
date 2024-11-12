#!/bin/bash
#SBATCH --job-name=inspect_data_inputs
#SBATCH --nodes=1
#SBATCH --ntasks=1          
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --time=00:30:00
#SBATCH --partition=dev_gpu_4_a100
#SBATCH --gres=gpu:1       

# Activate the environment
source activate lingua_241105

# Optional debugging configurations (uncomment if needed)
# NCCL_IB_DISABLE=1
# NCCL_BLOCKING_WAIT=1
# export NCCL_DEBUG=INFO  

# Run on a single GPU without torchrun
python -m apps.mtp.train config=apps/mtp/configs/debug.yaml

echo "MTP single-GPU job completed successfully 🏁."

