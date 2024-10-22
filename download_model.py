"""
Execute this during image build to avoid downloading models upon every docker run
"""
print("Executing download script")
from unsloth import FastLanguageModel
import os

hf_model_path = os.environ["MODEL_PATH"]
token = os.environ["HF_TOKEN"]

print(f"Will download this model from the hub: {hf_model_path}")

max_seq_length = 8192 # If input sequence length exceeds max_seq_len it will be capped to max_seq_len Llama has context length of 8192
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False # Whether to load the model in 4bit quantization. If True, uses 4bit quantization to reduce memory usage.


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = hf_model_path,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    token = token, # use one if using gated models like meta-llama/Llama-2-7b-hf
)

try:
    # Save locally
    save_in_local_folder = "model"
    if not os.path.exists(save_in_local_folder):
        os.makedirs(save_in_local_folder)
    model.save_pretrained_merged(save_in_local_folder, tokenizer, save_method="merged_16bit")
except Exception as e:
    print(f"Could not save locally due to error: {e}")
