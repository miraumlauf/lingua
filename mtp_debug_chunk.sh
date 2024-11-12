#!/bin/bash
#SBATCH --job-name=mtp_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8GB
#SBATCH --time=00:30:00
#SBATCH --partition=dev_gpu_4_a100
#SBATCH --gres=gpu:1

# Activate environment
source activate lingua_241105

# NCCL_IB_DISABLE=1
# NCCL_BLOCKING_WAIT=1
# export NCCL_DEBUG=INFO  
# Run Debug 
python -m apps.mtp.train config=apps/mtp/configs/debug.yaml
#torchrun --nproc-per-node=4 -m apps.mtp.train config=apps/mtp/configs/debug.yaml

echo "MTP debug job completed successfully 🏁."