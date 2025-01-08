import json
import argparse

def check_jsonl_file(file_path, output_errors_path):
    """
    Reads a JSONL file line by line, checks for JSON validity,
    and logs invalid lines into a separate file with problematic spans highlighted.
    
    Args:
        file_path (str): Path to the JSONL file.
        output_errors_path (str): Path to store problematic lines with errors.
    """
    invalid_lines = 0
    total_lines = 0

    with open(file_path, 'r') as infile, open(output_errors_path, 'w') as outfile:
        for line_num, line in enumerate(infile, start=1):
            total_lines += 1
            try:
                # Try parsing the JSON
                json.loads(line.strip())
            except json.JSONDecodeError as e:
                # If invalid, log the error and the line
                invalid_lines += 1
                start = e.pos  # Starting position of the problematic span
                end = start + 10 if start + 10 < len(line) else len(line)  # Expand span for better context
                
                error_message = f"Line {line_num}: {str(e)}"
                problematic_span = line[start:end]  # Extract problematic text
                print(f"Error on line {line_num}: {error_message}")
                print(f"Problematic span: {problematic_span}\n")
                
                # Write the error to the output file
                outfile.write(f"Line {line_num}:\n")
                outfile.write(f"Problematic line: {line}\n")
                outfile.write(f"Error: {error_message}\n")
                outfile.write(f"Problematic span: {problematic_span}\n\n")

    print(f"Total lines processed: {total_lines}")
    print(f"Invalid lines detected: {invalid_lines}")
    print(f"Problematic lines have been logged to: {output_errors_path}")


def main():
    # Argument parser for command-line arguments
    parser = argparse.ArgumentParser(description="Check JSONL file for errors.")
    parser.add_argument(
        "--file", required=True, help="Path to the JSONL file to check."
    )
    parser.add_argument(
        "--output", required=True, help="Path to save problematic lines with errors."
    )
    args = parser.parse_args()

    # Call the function with command-line arguments
    check_jsonl_file(args.file, args.output)


if __name__ == "__main__":
    main()
