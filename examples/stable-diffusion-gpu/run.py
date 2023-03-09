import os
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image

cache_path = "./cached_models/"
model_id = "runwayml/stable-diffusion-v1-5"


def generate_image(**inputs):
    prompt = inputs["prompt"]

    torch.backends.cuda.matmul.allow_tf32 = True

    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        model_id,
        use_auth_token=os.environ["HUGGINGFACE_API_KEY"],
        subfolder="scheduler",
        cache_dir=cache_path,
        solver_order=2,
        prediction_type="epsilon",
        thresholding=False,
        algorithm_type="dpmsolver++",
        solver_type="midpoint",
        denoise_final=True,
    )

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        revision="fp16",
        torch_dtype=torch.float16,
        cache_dir=cache_path,
        scheduler=scheduler,
        # Add your own auth token from Huggingface
        use_auth_token=os.environ["HUGGINGFACE_API_KEY"],
    ).to("cuda")

    with torch.inference_mode():
        with torch.autocast("cuda"):
            image = pipe(prompt, num_inference_steps=20, guidance_scale=7.5).images[0]

    print(f"Saved Image: {image}")
    image.save("output.png")


if __name__ == "__main__":
    prompt = "a renaissance style photo of elon musk"
    generate_image(prompt=prompt)
