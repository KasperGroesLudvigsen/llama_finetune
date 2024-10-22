print("Starting fine tuning process")
from unsloth import FastLanguageModel
import torch
import os
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from datasets import load_dataset
from codecarbon import EmissionsTracker

RANDOM_SEED = 42

print(f"GPU available: {torch.cuda.is_available()}")

token = os.environ["HF_TOKEN"]

n_samples = os.environ["N_SAMPLES"] # set to -1 if you want to use all samples

# e.g. "meta-llama/Llama-3.2-1B-Instruct"
# AI-Sweden-Models/Llama-3-8B-instruct is the llama model with the best rank
hf_model_path = os.environ["MODEL_PATH"]

dataset_path = os.environ["DATASET"]

# save to this HF repo, e.g. "ThatsGroes/Llama-3.2-1B-Instruct-Skolegpt"
repo = os.environ["SAVE_TO"]
print(f"Will push the trained model to: {repo}")

print(f"Will fine tune this model: {hf_model_path}")


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

print("model loaded")

rank = 64
model = FastLanguageModel.get_peft_model(
    model,
    r = rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128 # A lower rank means fewer parameters are being learned, leading to a more parameter-efficient method   
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = rank, # standard practice seems to be to set this to 16. Mlabonne says its usually set to 1-2x the rank
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = True,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

######### I THINK I NEED TO USE THE LLAMA CHAT TEMPLATE
# Code adapted from: https://colab.research.google.com/drive/17zEV0325xRQvDuSiOp8E4QB5vnK6atgK#scrollTo=VuuBTFiLs1lg
# If it does not work, replace this block with the code in dataloading_old.py
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
)

def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }
pass

dataset = load_dataset(dataset_path, split = "train")

# select subset if n_samples is larger than 0, e.g. for testing purposes
if n_samples > 0:
    dataset = dataset.shuffle(seed=RANDOM_SEED).select(range(n_samples))

alpaca_prompt = """Nedenfor er en instruktion, der beskriver en opgave sammen med et input, der giver yderligere kontekst. Skriv et svar, der besvarer opgaven.

### Instruktion:
{}

### Input:
{}

### Svar:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
# Edited to match key values in skolegpt instruct
def formatting_prompts_func(examples):
    instructions = examples["system_prompt"]
    inputs       = examples["question"]
    outputs      = examples["response"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

dataset = load_dataset("kobprof/skolegpt-instruct", split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True,)
##################################

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text", # the key of the dict returned in formatting_prompts_func()
    max_seq_length = max_seq_length,
    dataset_num_proc = 4, # Number of processes to use for processing the dataset. Only used when packing = False
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 1,
        warmup_steps = 5,
        num_train_epochs = 1, # Set this for 1 full training run. OpenAI's default is 4 https://community.openai.com/t/how-many-epochs-for-fine-tunes/7027/5
        #max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

# Run LLM fine tuning and track emissions
tracker = EmissionsTracker()
tracker.start()
try:
    trainer_stats = trainer.train()
finally:
    tracker.stop()

# Save to huggingface
try:
    model.push_to_hub_merged(repo, tokenizer, save_method="merged_16bit", token = token)
except Exception as e:
    print(f"Could not push to hub due to error: {e}")


#model.push_to_hub(repo, token = token) # Online saving
#tokenizer.push_to_hub(repo, token = token) # Online saving
#if True: model.push_to_hub_merged(repo, tokenizer, save_method = "merged_16bit", token = token)

#model.save_pretrained("lora_model") # Local saving
#tokenizer.save_pretrained("lora_model")
#model.push_to_hub("ThatsGroes/llama-test", token = token) # Online saving
#tokenizer.push_to_hub("ThatsGroes/llama-test", token = token) # Online saving

try:
    # Save locally
    save_in_local_folder = "model"
    if not os.path.exists(save_in_local_folder):
        os.makedirs(save_in_local_folder)
    model.save_pretrained_merged(save_in_local_folder, tokenizer, save_method="merged_16bit")
except Exception as e:
    print(f"Could not save locally due to error: {e}")
