import json
import os
from typing import List

def merge_episode_files(episode_numbers: List[int], output_file: str = "train.jsonl"):
    """
    Merge multiple episode JSONL files into a single training file.

    Args:
        episode_numbers: List of episode numbers to include (e.g., [1, 2, 3])
        output_file: Name of the output file (default: "train.jsonl")
    """

    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, output_file)

    merged_data = []
    files_processed = []
    files_not_found = []

    for ep_num in episode_numbers:
        filename = f"Ep.{ep_num}_final.jsonl"
        filepath = os.path.join(script_dir, filename)

        if not os.path.exists(filepath):
            files_not_found.append(filename)
            print(f"Warning: File '{filepath}' not found, skipping...")
            continue

        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                line_count = 0
                for line in file:
                    line = line.strip()
                    if line:  # Skip empty lines
                        try:
                            data = json.loads(line)
                            merged_data.append(data)
                            line_count += 1
                        except json.JSONDecodeError as e:
                            print(f"Warning: Invalid JSON in {filename} at line {line_count + 1}: {e}")
                            continue

                files_processed.append((filename, line_count))
                print(f"Processed '{filename}': {line_count} conversations")

        except Exception as e:
            print(f"Error reading '{filename}': {e}")
            continue

    # Write merged data to output file
    if merged_data:
        try:
            with open(output_path, 'w', encoding='utf-8') as outfile:
                for item in merged_data:
                    json.dump(item, outfile, ensure_ascii=False)
                    outfile.write('\n')

            print(f"\n✅ Successfully merged {len(merged_data)} conversations into '{output_path}'")
            print(f"Output file location: {os.path.abspath(output_path)}")
            print(f"Files processed: {len(files_processed)}")

            if files_not_found:
                print(f"Files not found: {files_not_found}")

        except Exception as e:
            print(f"Error writing to '{output_path}': {e}")
    else:
        print("No data to merge. Please check your files.")

# Configuration: Specify which episodes to include
EPISODES_TO_MERGE = [1, 6, 7, 8, 9, 10]  # Add or remove episode numbers as needed

if __name__ == "__main__":
    # You can modify this list to include only the episodes you want
    episodes = EPISODES_TO_MERGE

    # Show current working directory
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Starting merge process for episodes: {episodes}")
    print("-" * 50)

    merge_episode_files(episodes)

    print("-" * 50)
    print("Merge process completed!")