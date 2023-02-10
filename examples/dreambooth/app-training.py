"""
Workflow:

Training on a set of images 
1. Call API with input images (can upload to S3 or store on disk)
2. Handler takes images and runs `python run_dreambooth.py`, which starts training

Inference on a set of user images
1. Call API with userID and inference payload (if any?)
2. Pull down trained model, run SD inference on the user model --> reference the cache path in persistent volume for the user ID 
"""

import beam

# The environment your code will run on
app = beam.App(
    name="dreambooth-training",
    cpu=8,
    memory="32Gi",
    gpu=1,  # 1 GPU has 16 Gb of GPU memory. (This is separate from the computer memory above.)
    python_version="python3.8",
    python_packages="requirements.txt",
)

# Deploys function as async webhook
app.Trigger.Webhook(
    inputs={"user_id": beam.Types.String(), "image_urls": beam.Types.Json()},
    handler="run_dreambooth.py:train_dreambooth",
)

# Persistent volume to store cached model
app.Mount.SharedVolume(app_path="./dreambooth", name="dreambooth")
