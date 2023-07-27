from beam import App, Runtime, Image, Volume

from io import BytesIO

import numpy as np
import whisper
from whisper.audio import *

device = "cuda"

app = App(
    name="whisper",
    runtime=Runtime(
        cpu=4,
        memory="8Gi",
        gpu="T4",
        image=Image(
            python_packages=["git+https://github.com/openai/whisper.git"],
            commands=["apt-get update && apt-get install -y ffmpeg"],
        ),
    ),
    volumes=[Volume(path="/vol", name="vol")],
)


@app.rest_api(keep_warm_seconds=300)
def transcribe(audio: bytes) -> dict:
    """Transcript audio to text using Whisper.

    Args:
        audio (bytes): Audio file to transcribe.

    Returns:
        dict: Transcription of audio.
    """

    # transcribe audio, log, and return text
    model = whisper.load_model("small", device=device, download_root="/volumes/vol/")
    result = model.transcribe(load_audio(BytesIO(audio)))
    print(f'Result test: {result["text"]}')
    return {"output": result["text"]}


def load_audio(inp: BytesIO):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Adapted from:
    https://github.com/openai/whisper/blob/7858aa9c08d98f75575035ecd6481f462d66ca27/whisper/audio.py#L22

    Parameters
    ----------
    file: str
        The audio file to open
    sr: int
        The sample rate to resample the audio if necessary
    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    try:
        out, _ = (
            ffmpeg.input("pipe:", threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .run(
                cmd=["ffmpeg", "-nostdin"],
                capture_stdout=True,
                capture_stderr=True,
                input=inp.getbuffer(),
            )
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
