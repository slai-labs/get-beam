"""
### Stable Diffusion 3 medium on Beam ###

**Deploy it as an API**

**IMPORTANT**

You need to provide a secret named `HF_TOKEN`, and fill it with a user
access token from huggingface generated from https://huggingface.co/settings/tokens.
Then open [the model repository](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers)
and fill the form in order to get access. Otherwise you'll get an error

```
huggingface_hub.utils._errors.GatedRepoError: 401 Client Error
Access to model stabilityai/stable-diffusion-3-medium-diffusers is restricted. You must be authenticated to access it
```

beam deploy sd3med.py:generate_image
"""
from beam import App, Runtime, Image, Output, Volume

from diffusers import StableDiffusion3Pipeline
import torch

cache_path = "./models"

# The environment your app runs on
app = App(
    name="stable-diffusion3-medium",
    runtime=Runtime(
        cpu=1,
        memory="32Gi",
        gpu="A10G",
        image=Image(
            python_version="python3.8",
            python_packages=[
                "diffusers[torch]>=0.29.0",
                "transformers[sentencepiece]",
                "torch",
                "pillow",
                "accelerate",
                "safetensors",
                "xformers",
                "omegaconf",
                "peft"
            ],
        ),
    ),
    volumes=[
        Volume(name="models", path="./models")
    ],
)

# This runs once when the container first boots
def load_models():
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers", 
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
    prompt = inputs["prompt"]
    pipe = inputs["context"]

    image = pipe(
        prompt, 
        num_inference_steps=28, 
        guidance_scale=7.0,
    ).images[0]
    
    print(f"Saved Image: {image}")
    image.save("output.png")
