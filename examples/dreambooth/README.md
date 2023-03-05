# Dreambooth

This app has two components: an app for training, and an app for inference.

## Training a Custom Model

This endpoint will take a list of input images as URLs, and fine-tune Stable Diffusion on those images. It also takes a user ID, so that you can reference the specific fine-tuned model later on when you generate customized images.

Here's what this code does:

1. Call API with input images
2. Handler takes images and runs `python run_training.py`, which starts training

```python
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

# Webhook API will take two inputs:
# - user_id, to identify the user training their custom model
# - image_urls, a list of image URLs
# - class prompt, which is the **general** category of the thing being trained on (e.g. a person)
# - instance prompt, which is the **specific** thing being trained (e.g. a sks person)
app.Trigger.Webhook(
    inputs={
        "user_id": beam.Types.String(),
        "image_urls": beam.Types.Json(),
        "class_prompt": beam.Types.String(),
        "instance_prompt": beam.Types.String(),
    },
    handler="run_training.py:train_dreambooth",
)


# File path where we'll save the generated images
app.Output.File(path="output.png", name="image-output")

# Persistent volume to store cached model
app.Mount.SharedVolume(path="./dreambooth", name="dreambooth")
```

## Generating Custom Images

```json
{
    "user_id": "111111",
    "instance_prompt": "a photo of a sks toy",
    "class_prompt": "a photo of a toy",
    "image_urls": [
        "https://huggingface.co/datasets/valhalla/images/resolve/main/2.jpeg",
        "https://huggingface.co/datasets/valhalla/images/resolve/main/3.jpeg",
        "https://huggingface.co/datasets/valhalla/images/resolve/main/5.jpeg",
        "https://huggingface.co/datasets/valhalla/images/resolve/main/6.jpeg",
    ]
}
```

## Deploying

You'll deploy two API endpoints: one to trigger training on a set images, and another to generate images.

We'll first deploy the training app:

```cURL
beam deploy app-training.py
```

You'll see this appear in the dashboard.

Now, we'll deploy the inference app:

```cURL
beam deploy app-inference.py
```

## Calling the Training API

Now that the app is deployed, we can demonstrate the end-to-end workflow.

```curl
curl -X POST --compressed "https://api.beam.cloud/lnmfd" \
    -H 'Accept: */*' \
    -H 'Accept-Encoding: gzip, deflate' \
    -H 'Authorization: Basic [YOUR_AUTH_TOKEN]' \
    -H 'Connection: keep-alive' \
    -H 'Content-Type: application/json' \
    -d '{"user_id": "111111", "image_urls": "[\"https://huggingface.co/datasets/valhalla/images/resolve/main/2.jpeg\", \"https://huggingface.co/datasets/valhalla/images/resolve/main/3.jpeg\", \"https://huggingface.co/datasets/valhalla/images/resolve/main/4.jpeg\"]", "class_prompt": "a photo of a toy", "instance_prompt": "a photo of a sks toy"}'
```

This returns a task ID:

```json
{"task_id":"403f3a8e-503c-427a-8085-7d59384a2566"}
```

We can view the status of the training job by querying the `task` API:

```curl
curl -X POST --compressed "https://api.beam.cloud/task" \
  -H 'Accept: */*' \
  -H 'Accept-Encoding: gzip, deflate' \
  -H 'Authorization: Basic [YOUR_AUTH_TOKEN]' \
  -H 'Content-Type: application/json' \
  -d '{"action": "retrieve", "task_id": "403f3a8e-503c-427a-8085-7d59384a2566"}'
```

This returns the task status:

```json
{
   "outputs":{},
   "outputs_list":[],
   "started_at":"2023-02-15T22:26:11.941531Z",
   "ended_at":"2023-02-15T22:30:20.875621Z",
   "status":"COMPLETE",
   "task_id":"403f3a8e-503c-427a-8085-7d59384a2566"
}
```

## Calling the Inference API

```curl
curl -X POST --compressed "https://api.beam.cloud/lnmfd" \
    -H 'Accept: */*' \
    -H 'Accept-Encoding: gzip, deflate' \
    -H 'Authorization: Basic [YOUR_AUTH_TOKEN]' \
    -H 'Connection: keep-alive' \
    -H 'Content-Type: application/json' \
    -d '{"prompt": "photo of a sks riding the subway", "user_id": "111111"}'
```

```json
{
   "outputs":{
      "image-output":"https://beam.cloud/data/f2c8760c63d6e403729a212f1c19b597692b1c26c1c65"
   },
   "outputs_list":[
      {
         "id":"63ed62d4a6b28b22fbfd58bf",
         "created":"2023-02-15T22:55:16.347656Z",
         "name":"image-output",
         "updated":"2023-02-15T22:55:16.347674Z",
         "output_type":"file",
         "task":"403f3a8e-503c-427a-8085-7d59384a2566"
      }
   ],
   "started_at":"2023-02-15T22:54:43.156854Z",
   "ended_at":"2023-02-15T22:55:16.438379Z",
   "status":"COMPLETE",
   "task_id":"403f3a8e-503c-427a-8085-7d59384a2566"
}
```