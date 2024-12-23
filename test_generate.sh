#!/bin/bash
#SBATCH --job-name=test_gen_lr_1
#SBATCH --nodes=1
#SBATCH --ntasks=1          
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=00:15:00
#SBATCH --partition=dev_gpu_4_a100
#SBATCH --gres=gpu:1       

# Activate the environment
source activate lingua_241105



# Optional debugging configurations (uncomment if needed)
# NCCL_IB_DISABLE=1
# NCCL_BLOCKING_WAIT=1
# export NCCL_DEBUG=INFO  

# Run on a single GPU without torchrun
python -m apps.mtp.generate config=apps/mtp/configs/generate.yaml
#python -m apps.mtp.eval config=apps/mtp/configs/eval.yaml

echo "Test finished"