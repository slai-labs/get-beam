from beam import App, Runtime, Image, Output, Volume

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

# Cached model
cache_path = "./redpajama-incite-instruct"

# Huggingface model
model_id = "togethercomputer/RedPajama-INCITE-Instruct-3B-v1"


# The environment your code will run on
app = App(
    name="redpajama-incite-instruct-3b-v1",
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
            ],
        ),
    ),
    volumes=[
        Volume(path="./redpajama-incite-instruct", name="redpajama-incite-instruct")
    ],
)


@app.task_queue(outputs=[Output(path="redpajama-incite-instruct_output.txt")])
def run(**inputs):
    # Takes prompt from task queue
    prompt = inputs["prompt"]

    # Tokenize and define model
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, cache_dir=cache_path
    )
    model = model.to("cuda:0")

    # Generate output
    input = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = input.input_ids.shape[1]
    outputs = model.generate(
        **input,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
        top_p=0.7,
        top_k=50,
        return_dict_in_generate=True,
    )
    token = outputs.sequences[0, input_length:]
    generated_text = tokenizer.decode(token)

    # Display and save output
    print(generated_text)
    with open("redpajama-incite-instruct_output.txt", "w") as file:
        file.write(generated_text)
