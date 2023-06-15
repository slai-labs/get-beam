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

# This will be deployed as a task queue that can be triggered by a POST request
app.Trigger.TaskQueue(
    inputs={},
    handler="train.py:train",
)

# After deploying, you can trigger the task queue (you can get the URL from the Beam Dashboard)

# For any more information, please visit our docs, https://docs.beam.cloud and join our Slack!
