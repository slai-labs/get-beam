"""
### Stable Diffusion on Beam ###

**Deploy it as an API**

beam deploy app.py:generate_image
"""
from beam import App, Runtime, Image, Output, Volume

import os
import torch
from diffusers import StableDiffusionPipeline

cache_path = "./models"
model_id = "runwayml/stable-diffusion-v1-5"

# The environment your app runs on
app = App(
    name="stable-diffusion-app",
    runtime=Runtime(
        cpu=2,
        memory="16Gi",
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
    volumes=[Volume(name="models", path="./models")],
)

# This runs once when the container first boots
def load_models():
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        revision="fp16",
        torch_dtype=torch.float16,
        cache_dir=cache_path,
    ).to("cuda")

    return pipe


@app.task_queue(
    loader=load_models,
    outputs=[Output(path="output.png")],
    keep_warm_seconds=60,
)
def generate_image(**inputs):
    # Grab inputs passed to the API
    try:
        prompt = inputs["prompt"]
    # Use a default prompt if none is provided
    except KeyError:
        prompt = "a renaissance style photo of elon musk"
    
    # Retrieve pre-loaded model from loader
    pipe = inputs["context"]

    torch.backends.cuda.matmul.allow_tf32 = True

    with torch.inference_mode():
        with torch.autocast("cuda"):
            image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

    print(f"Saved Image: {image}")
    image.save("output.png")