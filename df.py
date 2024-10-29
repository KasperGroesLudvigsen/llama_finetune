import pandas as pd


# Replace 'path_to_file.jsonl' with the actual path to your JSONL file
df = pd.read_json('scandeval_benchmark_results_all.jsonl', lines=True)

# Display the first few rows of the DataFrame
print(df)

df.columns

df["results"][0]