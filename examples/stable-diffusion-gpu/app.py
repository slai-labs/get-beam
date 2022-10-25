import beam

app = beam.App(
    name="stable-diffusion",
    cpu=4,
    memory="16Gi",
    gpu=1,
    apt_install=[],
    python_version="python3.8",
    python_packages=["diffusers", "transformers", "torch", "pillow"],
)

app.Outputs.File(path="output.png", name="myimage")

app.Trigger.Webhook(
    inputs={"prompt": beam.Types.String()},
    outputs={},
    handler="run.py:generate_image",
)