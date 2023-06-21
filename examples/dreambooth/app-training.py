""" 
This app has two components:

1. Fine-tune Stable Diffusion (using Dreambooth) by supplying a user ID and list of image URLs
2. Generate images with Stable Diffusion by supplying a user ID and a custom prompt (e.g. 'my friend xyz eating a taco in los angeles')

Example API Request: 

After the API is deployed, you'll make requests like this:

curl -X POST --compressed "https://api.beam.cloud/[YOUR_APP_ID]" \
    -H 'Accept: */*' \
    -H 'Accept-Encoding: gzip, deflate' \
    -H 'Authorization: Basic [YOUR_AUTH_TOKEN]' \
    -H 'Connection: keep-alive' \
    -H 'Content-Type: application/json' \
    -d '{"user_id": "111111", "image_urls": "[\"https://huggingface.co/datasets/valhalla/images/resolve/main/2.jpeg\", \"https://huggingface.co/datasets/valhalla/images/resolve/main/3.jpeg\", \"https://huggingface.co/datasets/valhalla/images/resolve/main/4.jpeg\"]", "class_prompt": "a photo of a toy", "instance_prompt": "a photo of a sks toy"}'

"""


import beam

app = beam.App(
    name="dreambooth-training",
    gpu="A10G",
    cpu=8,
    memory="32Gi",
    python_version="python3.8",
    python_packages="requirements.txt",
)

# Deploys function as async task queue
app.Trigger.TaskQueue(
    inputs={
        "user_id": beam.Types.String(),
        "image_urls": beam.Types.Json(),
        "class_prompt": beam.Types.String(),
        "instance_prompt": beam.Types.String(),
    },
    handler="run_training.py:train_dreambooth",
)

# Shared Volume to store the trained models
app.Mount.SharedVolume(path="./dreambooth", name="dreambooth")
