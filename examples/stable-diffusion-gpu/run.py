import os
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from PIL import Image


def generate_image(**inputs):
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16,
        revision="fp16",
        # Add your personal auth token from Huggingface
        use_auth_token=os.environ["HUGGINGFACE_API_KEY"],
    ).to("cuda")

    prompt = inputs["prompt"]

    with autocast("cuda"):
        image = pipe(prompt, guidance_scale=7.5).images[0]
        print(image)

    image.save("output.png")
