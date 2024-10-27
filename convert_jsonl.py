import json

filename = "scandeval_benchmark_results_expanse"
# Input JSONL file path
jsonl_file = f"{filename}.jsonl"

# Output JSON file path
json_file = f"{filename}.json"

# Read the JSONL file and store each line as a list of dictionaries
data = []
with open(jsonl_file, 'r') as f:
    for line in f:
        data.append(json.loads(line))

# Write the list of dictionaries to a JSON file
with open(json_file, 'w') as f:
    json.dump(data, f, indent=4)

print(f"Converted {jsonl_file} to {json_file}")
