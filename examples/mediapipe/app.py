"""
This example is a basic implementation of MediaPipe.

Instructions for running this app:

1. Download the Beam CLI and create an account: https://docs.beam.cloud/getting-started/quickstart
2. Download this repo, and cd into the working directory. Run `beam start app.py`
3. Run `python run.py` to process the images in /images through MediaPipe

Next Steps:

You could deploy this app as an API by adding a Trigger. 

More info on Triggers here:
https://docs.beam.cloud/deployment/rest-api
https://docs.beam.cloud/deployment/task-queue

If you have any questions, just message us in Slack :)
"""

import beam

app = beam.App(
    name="mediapipe",
    cpu=8,
    memory="32Gi",
    python_packages=["opencv-python", "mediapipe"],
)

# Store unprocessed images in a Persistent Volume
app.Mount.PersistentVolume(name="unprocessed-images", app_path="unprocessed_images")

# Processed images will be saved as Output Files, which you can access in the web dashboard
app.Output.File(name="processed-image", path="output.png")