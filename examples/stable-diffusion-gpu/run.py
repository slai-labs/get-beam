import os

os.environ["TRANSFORMERS_CACHE"] = "/workspace/cached_models"
os.environ["HF_HOME"] = "/workspace/cached_models"

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from PIL import Image


def generate_image(**inputs):
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16,
        revision="fp16",
        use_auth_token=os.environ["HUGGINGFACE_API_KEY"],
    ).to("cuda")

    prompt = inputs["prompt"]

    with autocast("cuda"):
        image = pipe(prompt, guidance_scale=7.5).images[0]
        image.save("output.png")


if __name__ == "__main__":
    prompt = "a renaissance style portrait of steve jobs"
    generate_image(prompt=prompt)