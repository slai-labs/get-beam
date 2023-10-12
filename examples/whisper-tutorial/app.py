from beam import App, Runtime, Image, Volume, Output
import base64
import whisper
from whisper.audio import *

from tempfile import NamedTemporaryFile
import torch

device = "cuda"

app = App(
    name="whisper",
    runtime=Runtime(
        cpu=1,
        memory="8Gi",
        gpu="T4",
        image=Image(
            python_packages=["git+https://github.com/openai/whisper.git"],
            commands=["apt-get update && apt-get install -y ffmpeg"],
        ),
    ),
    volumes=[Volume(path="./cache", name="cache")],
)


def load_models():
    model = whisper.load_model("small", device=device, download_root="./cache")
    return model


@app.rest_api(outputs=[Output(path="output.txt")], loader=load_models)
def transcribe_audio(**inputs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Create a temporary file.
    with NamedTemporaryFile() as temp:
        # Write the user's uploaded file to the temporary file.
        audio_file = base64.b64decode(inputs["audio_file"].encode("utf-8"))
        temp.write(audio_file)

        # Retrieve model from loader
        model = inputs["context"]

        # Inference
        result = model.transcribe(temp.name)
        print(result["text"])

        # Write transcription to file output
        with open("output.txt", "w") as f:
            f.write(result["text"])

        return {"transcript": result["text"]}


if __name__ == "__main__":
    """'
    *** Testing Locally ***

    > beam start app.py
    > python app.py

    """
    import os

    mp3_filepath = os.path.abspath("example.wav")
    transcribe_audio(
        audio_file=base64.b64encode(open(mp3_filepath, "rb").read()).decode("UTF-8"),
        model="small",
    )
