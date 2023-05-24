import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

MIN_TRANSFORMERS_VERSION = '4.25.1'

# Check transformers version
assert transformers.__version__ >= MIN_TRANSFORMERS_VERSION, f'Please upgrade transformers to version {MIN_TRANSFORMERS_VERSION} or higher.'

# Cached model
cache_path = "./redpajama-incite-instruct"

# Huggingface model
model_id = "togethercomputer/RedPajama-INCITE-Instruct-3B-v1"

def run(**inputs):
    # Takes prompt from webhook
    prompt = inputs["prompt"]

    # Tokenize and define model
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        cache_dir=cache_path)
    model = model.to('cuda:0')
    
    # Generate output
    input = tokenizer(prompt, return_tensors='pt').to(model.device)
    input_length = input.input_ids.shape[1]
    inputs.pop("prompt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
        top_p=0.7,
        top_k=50,
        return_dict_in_generate=True)
    token = outputs.sequences[0, input_length:]
    generated_text = tokenizer.decode(token)
    
    # Display and save output
    print(generated_text)
    with open("redpajama-incite-instruct_output.txt", "w") as file:
        file.write(generated_text)
