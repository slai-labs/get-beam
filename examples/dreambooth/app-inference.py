"""
Running inference on a set of user images:
1. Calls API with userID and inference payload
2. Pulls down trained model and runs inference with stable diffusion

Example API Request: 

After the API is deployed, you'll make requests like this:

curl -X POST --compressed "https://api.beam.cloud/[YOUR_APP_ID]" \
    -H 'Accept: */*' \
    -H 'Accept-Encoding: gzip, deflate' \
    -H 'Authorization: Basic [YOUR_AUTH_TOKEN]' \
    -H 'Connection: keep-alive' \
    -H 'Content-Type: application/json' \
    -d '{"prompt": "a photo of a sks toy riding the subway", "user_id": "111111"}'

This API request will run asynchronously, and it will return a TaskID. 
You can retrieve the generated image through the /task API, by supplying the Task ID:

curl -X POST --compressed "https://api.beam.cloud/task" \
  -H 'Accept: */*' \
  -H 'Accept-Encoding: gzip, deflate' \
  -H 'Authorization: Basic [YOUR_AUTH_TOKEN]' \
  -H 'Content-Type: application/json' \
  -d '{"action": "retrieve", "task_id": "403f3a8e-503c-427a-8085-7d59384a2566"}'
"""


from beam import App, Runtime, Image, Output, Volume

import os
import torch
from diffusers import StableDiffusionPipeline

model_id = "runwayml/stable-diffusion-v1-5"


# The environment your code will run on
app = App(
    name="dreambooth-inference",
    runtime=Runtime(
        cpu=8,
        memory="32Gi",
        gpu="A10G",
        image=Image(
            python_version="python3.8",
            python_packages="requirements.txt",
        ),
    ),
    volumes=[Volume(path="./dreambooth", name="dreambooth")],
)


# TaskQueue API will take two inputs:
# - user_id, to identify the user training their custom model
# - image_urls, a list of image URLs
@app.task_queue(outputs=[Output(path="./dreambooth")])
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
