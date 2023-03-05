import os
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

model_id = "runwayml/stable-diffusion-v1-5"


def generate_images(**inputs):
    # Takes in a prompt and userID from the API request
    prompt = inputs["prompt"]
    user_id = inputs["user_id"]

    # Path to the unique model trained for this userID
    model_path = f"./dreambooth/trained_models/{user_id}"

    # Special torch method to improve performance
    torch.backends.cuda.matmul.allow_tf32 = True

    pipe = StableDiffusionPipeline.from_pretrained(
        # Run inference on the specific model trained for this user ID
        model_path,
        revision="fp16",
        torch_dtype=torch.float16,
        # The `cache_dir` arg is used to cache the model in between requests
        cache_dir=model_path,
    ).to("cuda")

    pipe.enable_xformers_memory_efficient_attention()

    # Image generation
    with torch.inference_mode():
        with torch.autocast("cuda"):
            image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

    print(f"Generated Image: {image}")
    image.save("output.png")


if __name__ == "__main__":
    user_id = "111111"
    generate_images(
        user_id=user_id,
        prompt=f"a photo of a sks toy riding the subway",
    )
