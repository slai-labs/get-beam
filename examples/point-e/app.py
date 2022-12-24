"""
In this example, we’ll demonstrate using Beam to deploy a cloud endpoint 
for OpenAI’s Point-E, a state-of-the-art model for generating 3D objects.
"""
import beam

# Define the environment
app = beam.App(
    name="pointe",
    cpu=4,
    memory="8Gi",
    gpu=1,
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
        "clip@git+https://github.com/openai/CLIP.git",
    ],
)

# Add a deployment trigger
app.Trigger.Webhook(
    inputs={"prompt": beam.Types.String()}, handler="run.py:generate_mesh"
)

# Add an Output path, to save generated 3D files
app.Output.File(path="mesh.ply", name="generated_mesh")