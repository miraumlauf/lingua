#!/bin/bash
#SBATCH --job-name=test_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1          
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=00:30:00
#SBATCH --partition=dev_gpu_4_a100
#SBATCH --gres=gpu:1       

# Activate the environment
source activate lingua_241105

module load devel/cuda/11.8



# Optional debugging configurations (uncomment if needed)
# NCCL_IB_DISABLE=1
# NCCL_BLOCKING_WAIT=1
# export NCCL_DEBUG=INFO  

# Run on a single GPU without torchrun
python -m apps.mtp.generate config=apps/mtp/configs/generate.yaml

echo "Test finished"