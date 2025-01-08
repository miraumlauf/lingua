#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

#SBATCH --job-name=scan_json
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=00:30:00
#SBATCH --partition=dev_single


source activate lingua_241105

#python setup/shuffle_babylm.py babylm_100M_clean_json 4 --data_dir ./babylm_data_json --seed 42
# python setup/shuffle_babylm.py babylm_dev_clean_json 4 --data_dir ./babylm_data_json --seed 42
# python setup/prepare_reduced_babylm_data.py babylm_10M 16 --data_dir ./babylm_data --seed 42
python setup/scan_json_files.py --file ./babylm_data_json/babylm_100M_clean_json/babylm_100M_clean_json.chunk.03.jsonl --output scan_output.txt

echo "BabyLM Data scanned"