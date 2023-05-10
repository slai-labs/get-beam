"""
A minimal example app which takes a Youtube URL as input and transcribes the video with OpenAI's Whisper.
"""

import beam

app = beam.App(
    name="whisper-example",
    cpu=8,
    memory="32Gi",
    gpu="A10G",
    python_version="python3.8",
    python_packages=[
        "git+https://github.com/openai/whisper.git",
        "pytube @ git+https://github.com/felipeucelli/pytube@03d72641191ced9d92f31f94f38cfb18c76cfb05",
    ],
    commands=["apt-get update && apt-get install -y ffmpeg"],
)

# This is deployed as a REST API, but for longer videos
# you'll want to deploy as an async Webhook instead, since the
# REST API has a 60s timeout
app.Trigger.RestAPI(
    inputs={"video_url": beam.Types.String()},
    outputs={"pred": beam.Types.String()},
    handler="run.py:transcribe",
)

# This is the file path where we'll save our downloaded videos
app.Output.File(name="video", path="video.mp3")