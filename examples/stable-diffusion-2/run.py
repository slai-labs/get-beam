import os

os.environ["TRANSFORMERS_CACHE"] = "/workspace/cached_models"
os.environ["HF_HOME"] = "/workspace/cached_models"

import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

from PIL import Image


def generate_images(**inputs):

    prompt = inputs["prompt"]

    model_id = "stabilityai/stable-diffusion-2"
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        scheduler=scheduler,
        revision="fp16",
        torch_dtype=torch.float16,
        use_auth_token=os.environ["HUGGINGFACE_API_KEY"],
    )
    pipe = pipe.to("cuda")

    image = pipe(prompt, height=768, width=768).images[0]
    image.save("output.png")


if __name__ == "__main__":
    prompt = "a photo of a hip hop artist knitting a piece of clothing"
    generate_images(prompt=prompt)
