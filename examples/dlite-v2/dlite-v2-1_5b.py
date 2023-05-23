import torch
from transformers import pipeline

# Cached model
cache_path = "./dlite-v2-1_5b"

# Huggingface model
model_id = "aisquared/dlite-v2-1_5b"

def run(**inputs):
    # Takes prompt from webhook
    prompt = inputs["prompt"]

    # Define the model
    pipe = pipeline(
        model=model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto")
    
    # Generate output
    generated_text = pipe(prompt)
    
    # Display and save output
    print(generated_text)
    with open("dlite_output.txt", "w") as file:
        file.write(generated_text)
