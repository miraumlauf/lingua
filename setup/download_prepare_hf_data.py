# Copyright (c) Meta Platforms, Inc. and affiliates.
# IMPORTANT: shuffeling does not work for streaming dataset.... Is done manually by using the following command
# cat data/fineweb_edu_streamed/fineweb_edu_sample.jsonl | data/fineweb_edu_streamed/terashuf/terashuf | split -n r/8 -d --suffix-length=2 --additional-suffix=.jsonl - data/fineweb_edu_streamed_shuffled/fineweb_edu.chunk.

import argparse
import os
import time
import subprocess
import requests
from huggingface_hub import snapshot_download
from itertools import islice  # NEW: Import for taking a limited number of samples
from datasets import load_dataset, Dataset  # NEW: Import for streaming from Hugging Face

def run_command(command):
    print(f"Running: {command}")
    subprocess.run(command, shell=True, check=True)


# def download_dataset(repo_id, local_dir, allow_patterns):
#     print(f"Downloading dataset from {repo_id}...")
#     max_retries = 5
#     retry_delay = 10  # seconds
#     for attempt in range(max_retries):
#         try:
#             snapshot_download(
#                 repo_id,
#                 repo_type="dataset",
#                 local_dir=local_dir,
#                 allow_patterns=allow_patterns,
#                 resume_download=True,
#                 max_workers=16, # Don't hesitate to increase this number to lower the download time
#             )
#             break
#         except requests.exceptions.ReadTimeout:
#             if attempt < max_retries - 1:
#                 print(f"Timeout occurred. Retrying in {retry_delay} seconds...")
#                 time.sleep(retry_delay)
#             else:
#                 raise
#     print(f"Dataset downloaded to {local_dir}")

    
# REPLACED the `download_dataset` function with `load_streamed_subset`
# This function streams a specified number of data points from the dataset
def load_streamed_subset(dataset_name, num_samples=1000):
    """Loads a small sample from the dataset via streaming and returns it as a Dataset object."""
    # Stream the dataset from Hugging Face
    streamed_dataset = load_dataset(dataset_name, split="train", streaming=True)
    # Take only the first `num_samples` entries using `islice`
    small_sample_list = list(islice(streamed_dataset, num_samples))
    # Convert to a Dataset object for compatibility with the rest of the code
    small_dataset = Dataset.from_list(small_sample_list)
    return small_dataset


def parquet_to_jsonl(dataset, work_dir, src_dir, tgt_dir, ntasks=64):
    from datatrove.executor import LocalPipelineExecutor
    from datatrove.pipeline.readers import ParquetReader
    from datatrove.pipeline.writers import JsonlWriter

    pipeline_exec = LocalPipelineExecutor(
        pipeline=[
            ParquetReader(
                src_dir,
                file_progress=True,
                doc_progress=True,
                glob_pattern="**/*.parquet",
            ),
            JsonlWriter(
                tgt_dir,
                output_filename=dataset + ".chunk.${rank}.jsonl",
                compression=None,
            ),
        ],
        tasks=ntasks,
        logging_dir=os.path.join(work_dir, "datatrove"),
    )
    pipeline_exec.run()


def setup_terashuf(work_dir):
    terashuf_dir = os.path.join(work_dir, "terashuf")
    terashuf_executable = os.path.join(terashuf_dir, "terashuf")

    if os.path.exists(terashuf_executable):
        print("terashuf executable already exists. Skipping setup.")
        return terashuf_dir

    print("Setting up terashuf...")
    run_command(f"git clone https://github.com/alexandres/terashuf {terashuf_dir}")
    run_command(f"make -C {terashuf_dir}")
    return terashuf_dir


def main(dataset, memory, data_dir, seed=42):
    # Configuration
    repo_id = {
        "fineweb_edu": "HuggingFaceFW/fineweb-edu",
        "fineweb_edu_10bt": "HuggingFaceFW/fineweb-edu",
        "dclm_baseline_1.0": "mlfoundations/dclm-baseline-1.0",
        "dclm_baseline_1.0_10prct": "mlfoundations/dclm-baseline-1.0",
    }[dataset]
    
    # CHANGED: Stream and prepare a small dataset sample instead of downloading everything
    print(f"Streaming and loading a small subset of {dataset}...")
    small_dataset = load_streamed_subset(repo_id, num_samples=1000)  # Use the streaming function to get a subset
    src_dir = f"{data_dir}/{dataset}_streamed"  # Updated source directory to reflect streamed data
    os.makedirs(src_dir, exist_ok=True)

    # CHANGED: Save the small dataset locally as JSONL for compatibility with the rest of the pipeline
    jsonl_path = os.path.join(src_dir, f"{dataset}_sample.jsonl")
    small_dataset.to_json(jsonl_path)  # Save the small sample to a JSONL file
    print(f"Saved a sample of the streamed dataset to {jsonl_path}")

    # ---- original start -----
    # src_dir = f"{data_dir}/{dataset}"
    # out_dir = f"{src_dir}_shuffled"
    # os.makedirs(out_dir, exist_ok=True)
    # --- orginal end ----
    
    
    work_dir = src_dir  # Directory of this Python file
    
    
    
    # ---- start original -----
    # prefix = f"{dataset}.chunk."
    # orig_extension = {
    #     "fineweb_edu": ".jsonl",
    #     "fineweb_edu_10bt": ".jsonl",
    #     "dclm_baseline_1.0": ".jsonl.zst",
    #     "dclm_baseline_1.0_10prct": ".jsonl.zst",
    # }[dataset]
    # cat_command = {
    #     "fineweb_edu": "cat",
    #     "fineweb_edu_10bt": "cat",
    #     "dclm_baseline_1.0": "zstdcat",
    #     "dclm_baseline_1.0_10prct": "zstdcat",
    # }[dataset]
    # allow_patterns = {
    #     "fineweb_edu": None,
    #     "fineweb_edu_10bt": "sample/10BT/*",
    #     "dclm_baseline_1.0": "*.jsonl.zst",
    #     "dclm_baseline_1.0_10prct": "global-shard_01_of_10/*.jsonl.zst",
    # }[dataset]
    
    #----- end original ------
    
    #CHANGED
    out_dir = f"{src_dir}_shuffled"
    os.makedirs(out_dir, exist_ok=True)
    prefix = f"{dataset}.chunk."
    
    # original again
    suffix = ".jsonl"
    nchunks = 5 # 38
    k_validation = 500 # 1000  # Number of lines to take from each chunk for validation

    # Setup terashuf
    terashuf_dir = setup_terashuf(work_dir)

    # # Download dataset
    # download_dataset(repo_id, src_dir, allow_patterns)

    # if "fineweb" in dataset:
    #     parquet_to_jsonl(dataset, work_dir, src_dir, src_dir)

    # Set up environment variables
    os.environ["MEMORY"] = f"{memory}"
    os.environ["SEED"] = f"{seed}"

    # # Run the original shuffling and splitting command
    # terashuf_executable = os.path.join(terashuf_dir, "terashuf")
    # run_command(
    #     f"ulimit -n 100000 && "
    #     f"find {src_dir} -type f -name '*{orig_extension}' -print0 | xargs -0 {cat_command} | {terashuf_executable} | "
    #     f"split -n r/{nchunks} -d --suffix-length 2 --additional-suffix {suffix} - {out_dir}/{prefix}"
    #     "; trap 'echo \"Caught signal 13, exiting with code 1\"; exit 1' SIGPIPE;"
    # )

    # CHANGED: Run the original shuffling and splitting command on the small streamed dataset
    terashuf_executable = os.path.join(terashuf_dir, "terashuf")
    run_command(
        f"ulimit -n 100000 && "
        f"cat {jsonl_path} | {terashuf_executable} | "
        f"split -n r/{nchunks} -d --suffix-length 2 --additional-suffix {suffix} - {out_dir}/{prefix}"
        "; trap 'echo \"Caught signal 13, exiting with code 1\"; exit 1' SIGPIPE;"
    )
    
    # Create validation set and remove lines from chunks
    validation_file = f"{out_dir}/{dataset}.val{suffix}"
    for i in range(nchunks):
        chunk_file = f"{out_dir}/{prefix}{i:02d}{suffix}"
        run_command(f"head -n {k_validation} {chunk_file} >> {validation_file}")
        run_command(f"sed -i '1,{k_validation}d' {chunk_file}")

    print("All tasks completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("memory", type=float, default=8)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    main(args.dataset, args.memory, args.data_dir, args.seed)
