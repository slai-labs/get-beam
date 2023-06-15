import beam

# The environment your code will run on
app = beam.App(
    name="dlite-v2-1_5b",
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

# Deploys function as a task queue
app.Trigger.TaskQueue(
    inputs={"prompt": beam.Types.String()},
    handler="dlite-v2-1_5b.py:run",
)

# File to store outputs
app.Output.File(path="dlite_output.txt", name="response")

# Persistent volume to store cached model
app.Mount.PersistentVolume(path="./dlite-v2-1_5b", name="dlite-v2-1_5b")
