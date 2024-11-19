import argparse
import os
import time
import subprocess
import json
import uuid

def run_command(command):
    print(f"Running: {command}")
    subprocess.run(command, shell=True, check=True)


def create_json_lines(input_file, output_file):
    """Convert plain text lines to JSONL format with only the 'text' key."""
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            json_object = {
                "text": line.strip()
            }
            outfile.write(json.dumps(json_object) + "\n")


def convert_train_to_jsonl(data_dir, jsonl_dir):
    """Convert all `.train` files in `data_dir` to JSONL in `jsonl_dir`."""
    os.makedirs(jsonl_dir, exist_ok=True)
    train_files = [f for f in os.listdir(data_dir) if f.endswith(".train")]
    for file in train_files:
        input_path = os.path.join(data_dir, file)
        output_path = os.path.join(jsonl_dir, file.replace(".train", ".jsonl"))
        create_json_lines(input_path, output_path)


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
    src_dir = os.path.join(data_dir, dataset)
    jsonl_dir = os.path.join(src_dir, "jsonl")
    out_dir = f"{src_dir}_shuffled"
    os.makedirs(out_dir, exist_ok=True)
    work_dir = os.path.dirname(__file__)
    prefix = f"{dataset}.chunk."
    suffix = ".jsonl"
    nchunks = 32

    # Convert `.train` files to `.jsonl`
    convert_train_to_jsonl(src_dir, jsonl_dir)

    # Setup terashuf
    terashuf_dir = setup_terashuf(work_dir)
    terashuf_executable = os.path.join(terashuf_dir, "terashuf")

    # Concatenate and shuffle JSONL files
    concatenated_file = os.path.join(jsonl_dir, "concatenated.jsonl")
    with open(concatenated_file, 'w') as outfile:
        for jsonl_file in os.listdir(jsonl_dir):
            if jsonl_file.endswith(".jsonl"):
                with open(os.path.join(jsonl_dir, jsonl_file), 'r') as infile:
                    outfile.write(infile.read())

    shuffled_file = os.path.join(jsonl_dir, "shuffled.jsonl")
    run_command(f"cat {concatenated_file} | {terashuf_executable} > {shuffled_file}")

    # Split into chunks
    run_command(
        f"split -n r/{nchunks} -d --suffix-length=2 --additional-suffix={suffix} {shuffled_file} {out_dir}/{prefix}"
    )

    print("All tasks completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("memory", type=float, help="Memory allocated for shuffling")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    main(args.dataset, args.memory, args.data_dir, args.seed)
