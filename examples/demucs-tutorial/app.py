from beam import App, Runtime, Image, Output, Volume, VolumeType
from pydub import AudioSegment

import os
import io
import subprocess
import requests


app = App(
    name="demucs-example",
    runtime=Runtime(
        cpu=1,
        memory="8Gi",
        gpu="T4",
        image=Image(
            python_version="python3.10",
            python_packages=[
                "demucs",
                "pydub",
            ],  # You can also add a path to a requirements.txt instead
            # Anything that would normally go inside a Dockerfile is in the commands field
            commands=[
                "apt-get update && apt-get install -y ffmpeg",
                "export TORCH_HOME='./model_weights'",
            ],
        ),
    ),
    # Mount a storage volume to cache model weights
    volumes=[
        Volume(
            name="model_weights",
            path="./model_weights",
            volume_type=VolumeType.Persistent,
        )
    ],
)


# This runs as an async task queue
@app.task_queue(
    # This function generates an output file which can be retrieved as a pre-signed URL
    outputs=[Output(path=f"processed/htdemucs/audio")]
)
def predict(**inputs):
    # Audio file URL retrieved from API
    audio_url = inputs["audio"]
    # Download the audio
    response = requests.request("GET", audio_url)
    audio = AudioSegment.from_file(io.BytesIO(response.content))
    audio.export("audio.wav")
    # Inference
    command = ["python", "-m", "demucs", "-d", "cuda", "audio.wav", "-o", "processed"]
    subprocess.run(command, env=dict(os.environ, TORCH_HOME="./model_weights"))


if __name__ == "__main__":
    predict(audio="https://www2.cs.uic.edu/~i101/SoundFiles/CantinaBand60.wav")
