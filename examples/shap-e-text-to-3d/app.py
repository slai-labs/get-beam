import beam

app = beam.App(
    name="shap-e-text-to-3d-example",
    cpu=8,
    memory="32Gi",
    gpu="A10G",
    python_version="python3.9",
    python_packages=[
        "filelock",
        "Pillow",
        "torch",
        "fire",
        "humanize",
        "requests",
        "tqdm",
        "matplotlib",
        "scikit-image",
        "scipy",
        "numpy",
        "blobfile",
        "clip @ git+https://github.com/openai/CLIP.git",
    ],
)


app.Trigger.Webhook(
    inputs={"prompt": beam.Types.String()},
    handler="run.py:generate_model",
    loader="run.py:load_models",
)

# This is the file path where we'll save our image
app.Output.File(path="output.png", name="image")

# This is the file path where we'll save our model
app.Output.File(path="model.ply", name="model")
