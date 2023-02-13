"""
Beam app
"""

import beam

app = beam.App(
    name="tacotron2",
    cpu=8,
    memory="16Gi",
    gpu="A10G",
    python_version=beam.types.PythonVersion.Python310,
    python_packages=[
        "numpy",
        "scipy",
        "pillow",
        "inflect",
        "librosa",
        "Unidecode",
        "torch",
        "inflect",
        "tqdm",
        "torchaudio",
        "speechbrain",
    ],
)

app.Trigger.RestAPI(
    inputs={"text": beam.Types.String()},
    outputs={"audio": beam.Types.String()},
    handler="run.py:to_speech",
    keep_warm_seconds=3000,
)
app.Mount.PersistentVolume(name="vol", app_path="/vol")
