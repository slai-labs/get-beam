"""
A minimal example app which takes a Youtube URL as input and transcribes the video with OpenAI's Whisper.
"""
from beam import App, Runtime, Image, Output, Volume

import os
import whisper
from pytube import YouTube


app = App(
    name="whisper-example",
    runtime=Runtime(
        cpu=8,
        memory="32Gi",
        gpu="A10G",
        image=Image(
            python_version="python3.8",
            python_packages=[
                "git+https://github.com/openai/whisper.git",
                "pytube @ git+https://github.com/felipeucelli/pytube@03d72641191ced9d92f31f94f38cfb18c76cfb05",
            ],
            commands=["apt-get update && apt-get install -y ffmpeg"],
        ),
    ),
)


# This is deployed as a REST API, but for longer videos
# you'll want to deploy as an async task queue instead, since the
# REST API has a 60s timeout
@app.rest_api(outputs=[Output(path="video.mp3")])
def transcribe(**inputs):
    # Grab the video URL passed from the API
    video_url = inputs["video_url"]

    # Create YouTube object
    yt = YouTube(video_url)
    video = yt.streams.filter(only_audio=True).first()

    # Download audio to the output path
    out_file = video.download(output_path="./")
    base, ext = os.path.splitext(out_file)
    new_file = base + ".mp3"
    os.rename(out_file, new_file)
    a = new_file

    # Load Whisper and transcribe audio
    model = whisper.load_model("small")
    result = model.transcribe(a)

    print(result["text"])
    return {"pred": result["text"]}


if __name__ == "__main__":
    video_url = "https://www.youtube.com/watch?v=adJFT6_j9Uk&ab_channel=minutephysics"
    transcribe(video_url=video_url)