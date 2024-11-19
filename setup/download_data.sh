#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

#SBATCH --job-name=download_babylm_data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=00:15:00
#SBATCH --partition=dev_single


source activate lingua_241105

python setup/prepare_reduced_babylm_data.py babylm_10M 16 --data_dir ./babylm_data --seed 42

echo "Data successfully downloaded"