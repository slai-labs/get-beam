"""
A minimal example app which takes a Youtube URL as input and transcribes the video with OpenAI's Whisper.
"""

import beam

app = beam.App(
    name="whisper-example",
    cpu=8,
    memory="4Gi",
    gpu=0,
    apt_install=[],
    python_version=beam.types.PythonVersion.Python38,
    python_packages=["git+https://github.com/openai/whisper.git", "youtube_dl"],
)

# This is deployed as a REST API, but for longer videos
# you'll want to deploy as an async Webhook instead, since the
# REST API has a 2 min timeout
app.Trigger.RestAPI(
    inputs={"video_url": beam.Types.String()},
    outputs={"pred": beam.Types.String()},
    handler="run.py:transcribe",
)

# This is the file path where we'll save our downloaded videos
app.Output.File(name="video", path="video.mp3")