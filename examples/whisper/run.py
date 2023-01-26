import whisper
import youtube_dl


def transcribe(**inputs):
    # Grab the video URL passed from the API
    video_url = inputs["video_url"]

    # Save the downloaded video to the Output defined in app.py
    download_path = "/workspace/video.mp3"
    options = {
        "format": "bestaudio/best",
        "keepvideo": False,
        "outtmpl": download_path,
    }

    # Download video to the download path defined above
    video_info = youtube_dl.YoutubeDL().extract_info(url=video_url, download=True)
    with youtube_dl.YoutubeDL(options) as ydl:
        ydl.download([video_info["webpage_url"]])

    # Load Whisper and transcribe audio
    model = whisper.load_model("small")
    result = model.transcribe(download_path)
    print(result["text"])

    return {"pred": result["text"]}


if __name__ == "__main__":
    video_url = "https://www.youtube.com/watch?v=adJFT6_j9Uk&ab_channel=minutephysics"
    transcribe(video_url=video_url)
