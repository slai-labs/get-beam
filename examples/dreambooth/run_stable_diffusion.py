import os
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image

cache_path = "/volumes/dreambooth/trained_models"
model_id = "runwayml/stable-diffusion-v1-5"


def generate_images(**inputs):
    prompt = inputs["prompt"]
    user_id = inputs["user_id"]

    torch.backends.cuda.matmul.allow_tf32 = True

    pipe = StableDiffusionPipeline.from_pretrained(
        # Run inference on the specific model trained for this user ID
        f"/volumes/dreambooth/trained_models/{user_id}",
        revision="fp16",
        torch_dtype=torch.float16,
        cache_dir=cache_path,
    ).to("cuda")

    pipe.enable_xformers_memory_efficient_attention()

    with torch.inference_mode():
        with torch.autocast("cuda"):
            image = pipe(prompt, num_inference_steps=20, guidance_scale=7.5).images[0]

    print(f"Generated Image: {image}")
    image.save("output.png")


if __name__ == "__main__":
    user_id = "12345"
    generate_images(user_id=user_id, prompt="man wearing sunglasses")