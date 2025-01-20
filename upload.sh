#!/bin/bash
#SBATCH --job-name=upload_test
#SBATCH --nodes=1
#SBATCH --ntasks=1          
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=00:05:00
#SBATCH --partition=dev_gpu_4_a100
#SBATCH --gres=gpu:1       

# Activate the environment
source activate lingua_241105



# Optional debugging configurations (uncomment if needed)
# NCCL_IB_DISABLE=1
# NCCL_BLOCKING_WAIT=1
# export NCCL_DEBUG=INFO  

# Run on a single GPU without torchrun
#python -m convert_llama_to_hf config=apps/mtp/configs/config_upload.yaml
python -m convert_lingua_to_hf_llama config=apps/mtp/configs/config_upload.yaml
#python -m test_generation

echo "Upload Test finished"