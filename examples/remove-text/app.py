"""
This app takes a base-64 encoded image as input and returns 
a modified image with all text removed. 

The app is deployed as a Beam Webhook. You can retrieve modified images through
the /task endpoint. 

Get started with our docs: docs.getbeam.dev
"""
import beam

app = beam.App(
    name="rmtext",
    cpu=4,
    memory="16Gi",
    python_packages=[
        "numpy",
        "matplotlib",
        "opencv-python",
        "keras_ocr",
        "tensorflow",
    ],
)

# Deploys app as a webhook, with a base64-encoded image as input
app.Trigger.Webhook(inputs={"image": beam.Types.String()}, handler="run.py:remove_text")

# Path to save generated images to
app.Output.File(path="output.png", name="image")