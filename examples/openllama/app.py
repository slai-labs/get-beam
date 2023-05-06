import beam

app = beam.App(
    name="openllama",
    cpu=16,
    memory="32Gi",
    gpu="A10G",
    python_packages=[
        "accelerate",
        "transformers",
        "torch",
        "sentencepiece",
    ],
)

# Deploy app as async webhook
app.Trigger.Webhook(
    inputs={"prompt": beam.Types.String()},
    handler="run.py:generate_text",
)

# Shared Volume to store cached model weights
app.Mount.SharedVolume(name="llama_weights", path="./llama_weights")
# Output file to store model output, since the task runs async
app.Output.File(name="output", path="output.txt")