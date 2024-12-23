#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

#SBATCH --job-name=create_dict_data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=180000MB
#SBATCH --time=10:00:00
#SBATCH --partition=single


source activate lingua_241105

python -m setup.create_dictionary

# python setup/prepare_reduced_babylm_data.py babylm_10M 16 --data_dir ./babylm_data --seed 42

echo "Data successfully processed with the json script ✔️"