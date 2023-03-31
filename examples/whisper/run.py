import os
import whisper
from pytube import YouTube


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
