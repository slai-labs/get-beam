import beam

app = beam.App(
    name="stable-diffusion-2",
    cpu=4,
    memory="16Gi",
    gpu=1,
    python_version="python3.9",
    python_packages=[
        "git+https://github.com/huggingface/diffusers.git",
        "transformers",
        "torch",
        "accelerate",
        "scipy",
        "pillow",
    ],
)

app.Trigger.Webhook(
    inputs={"prompt": beam.Types.String()}, handler="run.py:generate_images"
)

app.Output.File(path="output.png", name="generated_image")

app.Mount.PersistentVolume(app_path="./cached_models", name="cached_model")