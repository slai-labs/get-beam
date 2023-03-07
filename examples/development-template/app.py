"""
Get started with our docs, https://docs.beam.cloud
"""
import beam
from beam.types import GpuType, PythonVersion

app = beam.App(
    name="development-template",
    cpu=4,
    memory="16Gi",
    gpu=GpuType.T4,
    python_packages="requirements.txt",
    python_version=PythonVersion.Python38
)