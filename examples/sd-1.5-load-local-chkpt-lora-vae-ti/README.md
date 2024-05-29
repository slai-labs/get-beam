# Load local checkpoint and model weights on Beam

This example demonstrates how to load from local a checkpoint, lora, vae and ti for base model Stable Diffusion v1.5

Credits to [Talbo](https://x.com/TalboSocial)

## Overview

This app has an APIs to generate an image based on the prompt.

# Pre-requisites 

1. Make sure you have [Beam](https://beam.cloud) installed: `curl https://raw.githubusercontent.com/slai-labs/get-beam/main/get-beam.sh -sSfL | sh`
2. Clone this repo and `cd` into the directory

# Quickstart

0. Go to your Beam dashboard and upload the model weights on a Volume you create as described [here](https://docs.beam.cloud/data/volumes#uploading-files-with-the-dashboard).
Then make sure you add the path to each model weight in the constants section:
```
model_path = f"{volume_path}/Anything-V3.0-pruned-fp32.safetensors"
lora_path = f"{volume_path}/"
lora_name = "Crayon.safetensors"
ti_path = f"{volume_path}/1vanakn0ll.pt"
vae_path = f"{volume_path}/vae-ft-mse-840000-ema-pruned.safetensors"
```

1. Test the API Locally: `beam serve app.py`. You can make any desired changes to the code, and Beam will automatically 
  reload the remote server each time you update your application code. 
  Note: Any updates to compute requirements, python packages, or shell commands will require you to manually restart the dev session
2. Deploy the API to Beam: `beam deploy app.py`
  Once it's deployed, you can find the web URL in the dashboard.


## Calling the Inference API

Here's what a request will look like:

```curl

curl -X POST \
    --compressed 'https://uc6mc.apps.beam.cloud' \
    -H 'Accept: */*' \
    -H 'Accept-Encoding: gzip, deflate' \
    -H 'Authorization: Basic YOUR_AUTH_KEY' \
    -H 'Connection: keep-alive' \
    -H 'Content-Type: application/json' \
    -d '{"prompt": "photo of a girl riding the subway"}'
```
