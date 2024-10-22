# llama_finetune

# TODO:
### Run the docker image with a few samples and a test repo

### [V] What chat template was used for AI-swedens Llama instruct? 
I think this is solved ! 
I think I can use the code from this notebook: https://colab.research.google.com/drive/17zEV0325xRQvDuSiOp8E4QB5vnK6atgK#scrollTo=VuuBTFiLs1lg



Another option:
see here https://huggingface.co/AI-Sweden-Models/Llama-3-8B-instruct/discussions/5

# Hyperparameters to experiment with
- learning rate
- epochs
- rank

# What is LoRA?
"One cool property of low-rank matrices is that they can be represented as the product of two smaller matrices. This realization leads to the hypothesis that this delta between fine-tuned weights and initial pre-trained weights can be represented as the matrix product of two much smaller matrices. By focusing on updating these two smaller matrices rather than the entire original weight matrix, computational efficiency can be substantially improved. " https://www.anyscale.com/blog/fine-tuning-llms-lora-or-full-parameter-an-in-depth-analysis-with-llama-2


"Low-Rank Adaptation (LoRA) is a popular parameter-efficient fine-tuning technique. Instead of retraining the entire model, it freezes the weights and introduces small adapters (low-rank matrices) at each targeted layer. This allows LoRA to train a number of parameters that is drastically lower than full fine-tuning (less than 1%), reducing both memory usage and training time. This method is non-destructive since the original parameters are frozen, and adapters can then be switched or combined at will." https://mlabonne.github.io/blog/posts/2024-07-29_Finetune_Llama31.html


# FastLanguageModel.get_peft_model()
```
model = FastLanguageModel.get_peft_model(
    model,
    r = 64, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128 # A lower rank means fewer parameters are being learned, leading to a more parameter-efficient method   
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16, # standard practice seems to be to set this to 16
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)
```


## `r` (rank)
"Rank (r), which determines LoRA matrix size. Rank typically starts at 8 but can go up to 256. Higher ranks can store more information but increase the computational and memory cost of LoRA. We set it to 16 here." https://mlabonne.github.io/blog/posts/2024-07-29_Finetune_Llama31.html 

Rank in LoRA: The rank denotes the size of these low-rank matrices. A lower rank means fewer parameters are being learned, leading to a more parameter-efficient method. The rank can be thought of as a trade-off:

Higher Rank: More capacity to capture complex relationships and potentially better performance, but at the cost of increased parameters and training complexity.
Lower Rank: Fewer parameters, faster training, and less risk of overfitting, but it may limit the model’s ability to fully adapt to the new task.

"Choosing a higher rank for our decomposition matrices would counteract LoRA's efficiency gains. Our preliminary tests suggested minimal performance boosts when increasing the rank to, for instance, 16. As a result, we settled on a rank of 8 to maintain smaller checkpoint sizes and to avoid artificially inflating our checkpoint files." (https://www.anyscale.com/blog/fine-tuning-llms-lora-or-full-parameter-an-in-depth-analysis-with-llama-2)

## `target modules`
`print(model)` will show the modules of the model. 
For AI-Sweden/Llama-3.1-8b-instruct:
```
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(128256, 4096)
    (layers): ModuleList(
      (0-31): 32 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear4bit(in_features=4096, out_features=1024, bias=False)
          (v_proj): Linear4bit(in_features=4096, out_features=1024, bias=False)
          (o_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear4bit(in_features=4096, out_features=14336, bias=False)
          (up_proj): Linear4bit(in_features=4096, out_features=14336, bias=False)
          (down_proj): Linear4bit(in_features=14336, out_features=4096, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
      )
    )
    (norm): LlamaRMSNorm((4096,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=4096, out_features=128256, bias=False)
)
```

From HF: https://huggingface.co/docs/peft/en/package_reference/lora#peft.LoraConfig.target_modules 
"The names of the modules to apply the adapter to. If this is specified, only the modules with the specified names will be replaced. When passing a string, a regex match will be performed. When passing a list of strings, either an exact match will be performed or it is checked if the name of the module ends with any of the passed strings. If this is specified as ‘all-linear’, then all linear/Conv1D modules are chosen, excluding the output layer. If this is not specified, modules will be chosen according to the model architecture. If the architecture is not known, an error will be raised — in this case, you should specify the target modules manually."

The below blog chose to fine tune all dense layers:
"The original LoRA paper focused on fine-tuning only the "Q" and "V" attention matrices, achieving solid results that attested to the technique's efficacy. However, subsequent work has shown that targeting additional layers, or even all layers, can improve performance. We hypothesize that applying LoRA to a greater number of layers brings us closer to achieving the capabilities of full-parameter fine-tuning. Accordingly, we opted to implement LoRA across all layers."
https://www.anyscale.com/blog/fine-tuning-llms-lora-or-full-parameter-an-in-depth-analysis-with-llama-2#target-modules:-all-dense-layers



## `lora_alpha`
lora_alpha (int) — The alpha parameter for Lora scaling.

"Alpha (α), a scaling factor for updates. Alpha directly impacts the adapters’ contribution and is often set to 1x or 2x the rank value." https://mlabonne.github.io/blog/posts/2024-07-29_Finetune_Llama31.html

From https://datascience.stackexchange.com/questions/123229/understanding-alpha-parameter-tuning-in-lora-paper:

"In this blog (https://www.anyscale.com/blog/fine-tuning-llms-lora-or-full-parameter-an-in-depth-analysis-with-llama-2), they say:

'Alpha scales the learned weights. Existing literature, including the original LoRA paper, generally advises fixing Alpha—often at 16—rather than treating it as a tunable hyperparameter'

In literature, they say:

'[...] and LoRA alpha is the scaling factor for the weight matrices. The weight matrix is scaled by lora_alphalora_rank
, and a higher alpha value assigns more weight to the LoRA activations. We chose 16 since this was common practice in training scripts we reviewed and chose a 1:1 ratio so as not to overpower the base model.'

"

## `modules_to_save`
What value should this have?
The deafult value is None and neither Unsloth nor Alexandra Inst specify this.

## `use_rslora`
"In addition, we will use Rank-Stabilized LoRA (rsLoRA), which modifies the scaling factor of LoRA adapters to be proportional to 1/√r instead of 1/r. This stabilizes learning (especially for higher adapter ranks) and allows for improved fine-tuning performance as rank increases. Gradient checkpointing is handled by Unsloth to offload input and output embeddings to disk and save VRAM." https://mlabonne.github.io/blog/posts/2024-07-29_Finetune_Llama31.html

# SFTTrainer()
``` 
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text", # the key of the dict returned in formatting_prompts_func()
    max_seq_length = max_seq_length,
    dataset_num_proc = 4, # only used when packing = False
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 1,
        warmup_steps = 5,
        num_train_epochs = 1, # Set this for 1 full training run.
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

```
## `num_train_epochs`


## `max_seq_len`
It seems that if an input sequence exceeds max_seq_len the sequence will be capped to max_sew_len

"When loading the model, we must specify a maximum sequence length, which restricts its context window. Llama 3.1 supports up to 128k context length, but we will set it to 2,048 in this example since it consumes more compute and VRAM. "
https://mlabonne.github.io/blog/posts/2024-07-29_Finetune_Llama31.html


## `gradient_accumulation_steps`
"Gradient accumulation is a way to virtually increase the batch size during training, which is very useful when the available GPU memory is insufficient to accommodate the desired batch size. In gradient accumulation, gradients are computed for smaller batches and accumulated (usually summed or averaged) over multiple iterations instead of updating the model weights after every batch. Once the accumulated gradients reach the target “virtual” batch size, the model weights are updated with the accumulated gradients."
https://lightning.ai/blog/gradient-accumulation/

We set it to 1 as per Alexandra Inst.
## Learning rate
From: https://www.anyscale.com/blog/fine-tuning-llms-lora-or-full-parameter-an-in-depth-analysis-with-llama-2:
We further show that adopting a lower learning rate can enhance the reliability of the resulting model checkpoints. Experiments were carried out with a LoRA-adapted version of this script.

"Base learning rate: 1e-4
A learning rate of 1e-4 has become the standard when fine-tuning LLMs with LoRA. Although we occasionally encountered training loss instabilities, reducing the learning rate to lower values like 3e-5 proved effective in stabilizing the process—more on this will follow." https://www.anyscale.com/blog/fine-tuning-llms-lora-or-full-parameter-an-in-depth-analysis-with-llama-2#target-modules:-all-dense-layers

