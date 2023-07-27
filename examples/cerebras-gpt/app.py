from beam import App, Runtime, Image, Output, Volume

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Cached model
cache_path = "./cerebras-gpt"

# Huggingface model
model_id = "cerebras/Cerebras-GPT-1.3B"
device = "cuda"


# The environment your code will run on
app = App(
    name="cerebras-gpt",
    runtime=Runtime(
        cpu=16,
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
            ],
        ),
    ),
    volumes=[Volume(name="cerebras-gpt", path="./cerebras-gpt")],
)


@app.task_queue(outputs=[Output(path="cerebrasgpt_output.txt")])
def run(**inputs):
    # Takes prompt from task queue
    prompt = inputs["prompt"]

    # Tokenize and define model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_path).to(
        device
    )

    # Generate output
    input = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated_text = model.generate(input, max_length=60)
    generated_text = tokenizer.decode(generated_text[0])

    # Display and save output
    print(generated_text)
    with open("cerebrasgpt_output.txt", "w") as file:
        file.write(generated_text)
