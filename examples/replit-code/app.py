from beam import App, Runtime, Image, Output, Volume

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Cached model
cache_path = "./replit-code"

# Huggingface model
model_id = "replit/replit-code-v1-3b"
device = "cuda"


# The environment your code will run on
app = App(
    name="replit-code",
    runtime=Runtime(
        cpu=8,
        memory="32Gi",
        gpu="A10G",
        image=Image(
            python_version="python3.8",
            python_packages=[
                "diffusers[torch]>=0.10",
                "transformers",
                "torch",
                "pillow",
                "accelerate",
                "safetensors",
                "xformers",
                "sentencepiece",
                "einops",
            ],
        ),
    ),
    volumes=[Volume(path="./replit-code", name="replit-code")],
)


@app.task_queue(outputs=[Output(path="replit-code_output.txt")])
def run(**inputs):
    # Takes prompt from task queue
    prompt = inputs["prompt"]

    # Tokenize and define model
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, cache_dir=cache_path
    )
    model.to(device="cuda:0", dtype=torch.bfloat16)

    # Generate output
    input = tokenizer.encode(prompt, return_tensors="pt")
    input = input.to(device="cuda:0")
    generated_code = model.generate(
        input,
        max_length=100,
        do_sample=True,
        top_p=0.95,
        top_k=4,
        temperature=0.2,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated_code = tokenizer.decode(
        generated_code[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    # Display and save output
    print(generated_code)
    with open("replit-code_output.txt", "w") as file:
        file.write(generated_code)
