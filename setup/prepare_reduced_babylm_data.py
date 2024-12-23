import argparse
import os
import subprocess
import json

# Execute bash commands
def run_command(command):
    print(f"Running: {command}")
    subprocess.run(command, shell=True, check=True)

# Shuffle the dataset using terashuf
def shuffle_dataset(data_dir, dataset, work_dir):
    """Shuffle the dataset using terashuf."""
    concatenated_file = os.path.join(data_dir, f"{dataset}_concatenated.train")
    shuffled_file = os.path.join(data_dir, f"{dataset}_shuffled.train")

    # Concatenates all `.train` files
    with open(concatenated_file, 'w') as outfile:
        for file in os.listdir(os.path.join(data_dir, dataset)):
            if file.endswith(".train"):
                with open(os.path.join(data_dir, dataset, file), 'r') as infile:
                    outfile.write(infile.read())

    # Shuffle the concatenated file
    terashuf_dir = setup_terashuf(work_dir)
    terashuf_executable = os.path.join(terashuf_dir, "terashuf")
    run_command(f"cat {concatenated_file} | {terashuf_executable} > {shuffled_file}")

    return shuffled_file

# Reduce the dataset to 2 million words
def reduce_dataset_to_words(shuffled_file, max_words=2000000):
    """Reduce the shuffled dataset to `max_words`."""
    reduced_file = shuffled_file.replace("_shuffled.train", "_reduced.train")
    total_words = 0

    with open(shuffled_file, 'r') as infile, open(reduced_file, 'w') as outfile:
        for line in infile:
            words = line.split()
            word_count = len(words)

            # If adding the current line exceeds max_words, take only the required portion
            if total_words + word_count > max_words:
                remaining_words = max_words - total_words
                outfile.write(' '.join(words[:remaining_words]) + '\n')
                break

            # Otherwise, write the entire line
            outfile.write(line)
            total_words += word_count

    return reduced_file

# Convert plain text lines to JSONL format
def create_json_lines(input_file, output_file):
    """Convert plain text lines to JSONL format with only the 'text' key."""
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            json_object = {
                "text": line.strip()
            }
            outfile.write(json.dumps(json_object) + "\n")

# Convert reduced `.train` file to JSONL
def convert_train_to_jsonl(input_file, jsonl_dir):
    """Convert a single reduced `.train` file to JSONL in `jsonl_dir`."""
    os.makedirs(jsonl_dir, exist_ok=True)
    output_path = os.path.join(jsonl_dir, "reduced.jsonl")  # Fixed output name for reduced data
    create_json_lines(input_file, output_path)

# Setup terashuf for shuffling
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

# Split the shuffled dataset into chunks
def split_dataset(shuffled_file, out_dir, dataset, nchunks=32):
    """Split the shuffled dataset into equal-sized chunks."""
    prefix = f"{dataset}.chunk."
    suffix = ".jsonl"

    # Ensure the output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Split the shuffled file into chunks
    run_command(
        f"split -n r/{nchunks} -d --suffix-length=2 --additional-suffix={suffix} {shuffled_file} {out_dir}/{prefix}"
    )

# Main function to manage the pipeline
def main(dataset, memory, data_dir, seed=42, nchunks=32):
    # Configuration
    src_dir = os.path.join(data_dir, dataset)
    jsonl_dir = os.path.join(src_dir, "jsonl")
    out_dir = f"{src_dir}_shuffled"
    os.makedirs(out_dir, exist_ok=True)
    work_dir = os.path.dirname(__file__)

    # Step 1: Shuffle the dataset
    shuffled_file = shuffle_dataset(data_dir, dataset, work_dir)

    # Step 2: Reduce the shuffled dataset to 2 million words
    reduced_file = reduce_dataset_to_words(shuffled_file, max_words=2000000)

    # Step 3: Convert the reduced dataset to JSONL format
    convert_train_to_jsonl(reduced_file, jsonl_dir)

    # Step 4: Split the shuffled dataset into chunks
    split_dataset(reduced_file, out_dir, dataset, nchunks)

    print("All tasks completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("memory", type=float, help="Memory allocated for shuffling")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nchunks", type=int, default=32)

    args = parser.parse_args()

    main(args.dataset, args.memory, args.data_dir, args.seed, args.nchunks)
