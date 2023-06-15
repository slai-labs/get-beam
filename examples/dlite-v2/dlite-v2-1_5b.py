import torch
from instruct_pipeline import InstructionTextGenerationPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

# Cached model
cache_path = "./dlite-v2-1_5b"

# Huggingface model
model_id = "aisquared/dlite-v2-1_5b"

def run(**inputs):
    # Takes prompt from task queue
    prompt = inputs["prompt"]
 
    # Define the model
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        cache_dir=cache_path)
    
    # Generate output
    generate_text = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)
    generated_text = generate_text(prompt)
    
    # Display and save output
    print(generated_text)
    with open("dlite_output.txt", "w") as file:
        file.write(generated_text)
