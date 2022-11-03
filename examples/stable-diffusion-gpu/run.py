import os
from dotenv import load_dotenv
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from PIL import Image

load_dotenv()


def generate_image(**inputs):
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16,
        revision="fp16",
        use_auth_token=os.getenv("HUGGING_FACE_TOKEN"),
    ).to("cuda")

    prompt = inputs["prompt"]

    with autocast("cuda"):
        image = pipe(prompt, guidance_scale=7.5).images[0]
        print(image)

        image.save("output.png")


if __name__ == "__main__":
    prompt = "a renaissance style portrait of steve jobs"
    generate_image(prompt=prompt)