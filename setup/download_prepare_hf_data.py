# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import os
import time
import subprocess
from itertools import islice
from datasets import load_dataset, Dataset

def run_command(command):
    """Runs a shell command and ensures it completes successfully."""
    print(f"Running: {command}")
    subprocess.run(command, shell=True, check=True)


def load_streamed_subset(dataset_name, num_samples=5000):
    """
    Loads a small sample from the dataset via streaming and returns it as a Dataset object.
    """
    print(f"Streaming dataset: {dataset_name}...")
    streamed_dataset = load_dataset(dataset_name, split="train", streaming=True)
    # Take only the first `num_samples` entries using islice
    small_sample_list = list(islice(streamed_dataset, num_samples))
    # Convert to a Dataset object for compatibility
    small_dataset = Dataset.from_list(small_sample_list)
    return small_dataset


def setup_terashuf(work_dir):
    """Sets up the terashuf tool for shuffling large text datasets."""
    terashuf_dir = os.path.join(work_dir, "terashuf")
    terashuf_executable = os.path.join(terashuf_dir, "terashuf")

    if os.path.exists(terashuf_executable):
        print("terashuf executable already exists. Skipping setup.")
        return terashuf_dir

    print("Setting up terashuf...")
    run_command(f"git clone https://github.com/alexandres/terashuf {terashuf_dir}")
    run_command(f"make -C {terashuf_dir}")
    return terashuf_dir


def main(dataset, memory, data_dir, seed=42, nchunks=32):
    # Configuration
    repo_id = {
        "fineweb_edu": "HuggingFaceFW/fineweb-edu",
        "fineweb_edu_10bt": "HuggingFaceFW/fineweb-edu",
        "dclm_baseline_1.0": "mlfoundations/dclm-baseline-1.0",
        "dclm_baseline_1.0_10prct": "mlfoundations/dclm-baseline-1.0",
    }[dataset]

    # Prepare directories
    src_dir = f"{data_dir}/{dataset}_streamed"  # Updated for streaming data
    out_dir = f"{src_dir}_shuffled"
    os.makedirs(src_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    work_dir = src_dir  # Directory for intermediate files (e.g., terashuf)
    prefix = f"{dataset}.chunk."
    suffix = ".jsonl"
    k_validation = 10000  # Number of lines to take from each chunk for validation

    # Stream dataset and save locally as JSONL
    print(f"Streaming and preparing a small dataset sample for {dataset}...")
    small_dataset = load_streamed_subset(repo_id, num_samples=5000)
    jsonl_path = os.path.join(src_dir, f"{dataset}_sample.jsonl")
    small_dataset.to_json(jsonl_path)  # Save the streamed sample to a JSONL file
    print(f"Streamed dataset saved to {jsonl_path}.")

    # Setup terashuf
    terashuf_dir = setup_terashuf(work_dir)

    # Set up environment variables
    os.environ["MEMORY"] = f"{memory}"
    os.environ["SEED"] = f"{seed}"

    # Shuffle and split streamed dataset
    terashuf_executable = os.path.join(terashuf_dir, "terashuf")
    run_command(
        f"ulimit -n 100000 && "
        f"cat {jsonl_path} | {terashuf_executable} | "
        f"split -n r/{nchunks} -d --suffix-length 2 --additional-suffix {suffix} - {out_dir}/{prefix}"
        "; trap 'echo \"Caught signal 13, exiting with code 1\"; exit 1' SIGPIPE;"
    )

    print("All tasks completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="Name of the dataset to process")
    parser.add_argument("memory", type=float, help="Memory limit for shuffling")
    parser.add_argument("--data_dir", type=str, default="data", help="Base directory for data storage")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--nchunks", type=int, default=32, help="Number of chunks to split the dataset into")

    args = parser.parse_args()

    main(args.dataset, args.memory, args.data_dir, args.seed)
