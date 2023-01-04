import beam

# The environment your code will run on
app = beam.App(
    name="stable-diffusion-app",
    cpu=8,
    memory="32Gi",
    gpu=1,
    python_version="python3.8",
    python_packages=[
        "diffusers[torch]>=0.10",
        "transformers",
        "torch",
        "pillow",
        "triton",
        "accelerate",
        "xformers==0.0.16rc393",
        "safetensors",
    ],
)

# Deploys function as async webhook
app.Trigger.Webhook(
    inputs={"prompt": beam.Types.String()},
    handler="run.py:generate_image",
)

# File to store image outputs
app.Output.File(path="output.png", name="myimage")

# Persistent volume to store cached model
app.Mount.PersistentVolume(app_path="./cached_models", name="cached_model")