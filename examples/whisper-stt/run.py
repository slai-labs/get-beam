"""Main beam handler to do whisper stt inference"""

from io import BytesIO

import ffmpeg
import numpy as np
import whisper
from whisper.audio import *

device = "cuda"
model = whisper.load_model(
    "small", device=device, download_root="/volumes/vol/"
)

def load_audio(inp: BytesIO, sr: int = SAMPLE_RATE):
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


def transcribe(audio: bytes) -> dict:
    """Transcript audio to text using Whisper.

    Args:
        audio (bytes): Audio file to transcribe.

    Returns:
        dict: Transcription of audio.
    """

    # transcribe audio, log, and return text
    result = model.transcribe(load_audio(BytesIO(audio)))
    print(f'Result test: {result["text"]}')
    return {"output": result["text"]}
