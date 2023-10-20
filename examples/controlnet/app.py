from beam import App, Runtime, Image, Volume, RequestLatencyAutoscaler, Output

import sys

from diffusers import (
    StableDiffusionInpaintPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)

from diffusers.utils import load_image

import cv2
from PIL import Image as Img
import numpy as np
import torch


# Beam Volume to store cached models
CACHE_PATH = "./cached_models"

app = App(
    name="controlnet",
    runtime=Runtime(
        cpu=2,
        memory="32Gi",
        gpu="A10G",
        image=Image(
            python_version="python3.9",
            python_packages=[
                "diffusers==0.14.0",
                "xformers",
                "transformers",
                "scipy",
                "ftfy",
                "accelerate",
                "controlnet_aux",
            ],  # You can also add a path to a requirements.txt instead
            commands=[
                "apt-get update && git clone https://github.com/mikonvergence/ControlNetInpaint"
            ],
        ),
    ),
    # Storage Volume for model weights
    volumes=[Volume(name="cached_models", path=CACHE_PATH)],
)

sys.path.append("./ControlNetInpaint/")


# This runs as an async task queue - the output image can be retrieved by querying the /task API
# https://docs.beam.cloud/data/outputs#task-api
@app.task_queue(outputs=[Output(path="image.png")])
def predict(**inputs):
    # Inputs can be retrieved from the API like inputs["prompt"]

    pipe_sd = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        revision="fp16",
        torch_dtype=torch.float16,
        cache_dir=CACHE_PATH,
    )
    # speed up diffusion process with faster scheduler and memory optimization
    pipe_sd.scheduler = UniPCMultistepScheduler.from_config(pipe_sd.scheduler.config)
    # remove following line if xformers is not installed
    pipe_sd.enable_xformers_memory_efficient_attention()

    pipe_sd.to("cuda")

    # download an image
    image = load_image(
        "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
    )
    image = np.array(image)
    mask_image = load_image(
        "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"
    )
    mask_image = np.array(mask_image)

    text_prompt = "a red panda sitting on a bench"

    # load control net and stable diffusion v1-5
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny",
        torch_dtype=torch.float16,
        cache_dir=CACHE_PATH,
    )
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        cache_dir=CACHE_PATH,
    )

    # speed up diffusion process with faster scheduler and memory optimization
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    # remove following line if xformers is not installed
    pipe.enable_xformers_memory_efficient_attention()

    pipe.to("cuda")

    # get canny image
    canny_image = cv2.Canny(image, 100, 200)
    canny_image = canny_image[:, :, None]
    canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)

    image = Img.fromarray(image)
    mask_image = Img.fromarray(mask_image)
    canny_image = Img.fromarray(canny_image)

    # generate image
    generator = torch.manual_seed(0)
    img_response = pipe(
        text_prompt,
        num_inference_steps=20,
        generator=generator,
        image=image,
        # control_image=canny_image,
        # controlnet_conditioning_scale=0.5,
        mask_image=mask_image,
    ).images[0]

    img_response.save("image.png")
