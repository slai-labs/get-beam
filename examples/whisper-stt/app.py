"""
Beam app for doing whisper stt
"""

import beam

app = beam.App(
    name="whisper",
    cpu=8,
    memory="16Gi",
    gpu=1,
    apt_install=[],
    python_version=beam.types.PythonVersion.Python38,
    python_packages=["git+https://github.com/openai/whisper.git"],
)

app.Trigger.RestAPI(
    inputs={"audio": beam.Types.Binary()},
    outputs={"output": beam.Types.String()},
    handler="run.py:transcribe",
    keep_warm_seconds=3000,
)
app.Mount.PersistentVolume(name="vol", app_path="/vol")
