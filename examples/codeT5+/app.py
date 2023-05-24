import beam

# The environment your code will run on
app = beam.App(
    name="codet5p",
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
    handler="codeT5+.py:run",
)

# File to store outputs
app.Output.File(path="codet5p_output.txt", name="response")

# Persistent volume to store cached model
app.Mount.PersistentVolume(path="./codet5p", name="codet5p")
