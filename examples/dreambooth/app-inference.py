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


import beam

# The environment your code will run on
app = beam.App(
    name="dreambooth-inference",
    cpu=8,
    memory="32Gi",
    gpu="A10G",
    python_version="python3.8",
    python_packages="requirements.txt",
)

# TaskQueue API will take two inputs:
# - user_id, to identify the user training their custom model
# - image_urls, a list of image URLs
app.Trigger.TaskQueue(
    inputs={"user_id": beam.Types.String(), "prompt": beam.Types.String()},
    handler="run_inference.py:generate_images",
)

# File path where we'll save the generated images
app.Output.File(path="output.png", name="image-output")

# Shared Volume to store the trained models
app.Mount.SharedVolume(path="./dreambooth", name="dreambooth")
