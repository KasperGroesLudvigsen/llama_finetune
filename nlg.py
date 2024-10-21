import pandas as pd

df = pd.read_csv("danish-nlg.csv")

df[(df["model_id"].str.contains("llama")) | (df["model_id"].str.contains("Llama"))][["model_id", "rank"]]