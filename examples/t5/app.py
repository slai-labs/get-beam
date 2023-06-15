import beam

# The environment your code will run on
app = beam.App(
    name="t5",
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
        "sentencepiece",
    ],
)

# Deploys function as a task queue
app.Trigger.TaskQueue(
    inputs={"prompt": beam.Types.String()},
    handler="t5.py:run",
)

# File to store outputs
app.Output.File(path="t5_output.txt", name="response")

# Persistent volume to store cached model
app.Mount.PersistentVolume(path="./t5", name="t5")
