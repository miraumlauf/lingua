#!/bin/bash
#SBATCH --job-name=llama_4fh_128
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=00:30:00
#SBATCH --partition=dev_gpu_4_a100
#SBATCH --gres=gpu:4

# Activate environment
source activate lingua_241105

#NCCL_IB_DISABLE=1
#NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG=INFO  

export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Run Debug 
#python -m apps.mtp.train config=apps/mtp/configs/debug.yaml
# torchrun --nproc-per-node=4 -m apps.mtp.train config=apps/mtp/configs/llama_lr_8e-3.yaml
#torchrun --nproc-per-node=4 -m apps.mtp.train config=apps/mtp/llama_384/config.yaml
torchrun --nproc-per-node=4 -m apps.mtp.train config=apps/mtp/configs/llama_4fh_128.yaml

### ntp with main transformer
#torchrun --nproc-per-node=4 -m apps.main.train config=apps/main/configs/ntp_llama.yaml


echo "Change to config in dump dir! ✔️"
