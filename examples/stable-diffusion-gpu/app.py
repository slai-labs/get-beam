import beam

app = beam.App(
    name="stable-diffusion-video-example",
    cpu=4,
    gpu=1,
    memory="16Gi",
    python_version="python3.8",
    python_packages=["diffusers", "transformers", "torch", "pillow", "python-dotenv"],
)

app.Trigger.Webhook(
    inputs={"prompt": beam.Types.String()}, handler="run.py:generate_image"
)

app.Output.File(path="output.png", name="my_image")