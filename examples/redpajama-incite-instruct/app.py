import beam

# The environment your code will run on
app = beam.App(
    name="redpajama-incite-instruct-3b-v1",
    cpu=16,
    memory="32Gi",
    gpu="A10G",
    python_version="python3.8",
    python_packages=[
        "diffusers[torch]>=0.10",
        "transformers",
        "torch",
        "pillow",
        "accelerate",
        "safetensors",
        "xformers",
    ],
)

# Deploys function as async webhook
app.Trigger.Webhook(
    inputs={"prompt": beam.Types.String()},
    handler="redpajama-incite-instruct-3b-v1.py:run",
)

# File to store outputs
app.Output.File(path="redpajama-incite-instruct_output.txt", name="response")

# Shared volume to store cached model
app.Mount.PersistentVolume(path="./redpajama-incite-instruct", name="redpajama-incite-instruct")
