from beam import App, Runtime, Image, Output, Volume

import torch
from instruct_pipeline import InstructionTextGenerationPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

# Cached model
cache_path = "./dlite-v2-1_5b"

# Huggingface model
model_id = "aisquared/dlite-v2-1_5b"


# The environment your code will run on
app = App(
    name="dlite-v2-1_5b",
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
    volumes=[Volume(path="./dlite-v2-1_5b", name="dlite-v2-1_5b")],
)


@app.task_queue(outputs=[Output(path="dlite_output.txt")])
def run(**inputs):
    # Takes prompt from task queue
    prompt = inputs["prompt"]

    # Define the model
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.bfloat16, cache_dir=cache_path
    )

    # Generate output
    generate_text = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)
    generated_text = generate_text(prompt)

    # Display and save output
    print(generated_text)
    with open("dlite_output.txt", "w") as file:
        file.write(generated_text)
