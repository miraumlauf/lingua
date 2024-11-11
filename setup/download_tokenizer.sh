#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

#SBATCH --job-name=download_tokenizer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=8GB
#SBATCH --time=00:10:00
#SBATCH --partition=single

source activate lingua_241105

export HF_API_KEY="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"

# Run the download command and check if it succeeds
python setup/download_tokenizer.py llama3 ./tokenizers/llama3 --api_key $HF_API_KEY && \
echo "Tokenizer download completed successfully." || \
echo "Tokenizer download failed."