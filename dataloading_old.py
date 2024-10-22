
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
