from beam import App, Runtime, Image, Volume


from io import BytesIO
import base64


import torch
from scipy.io.wavfile import write

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False

device = "cuda"
volume_path = "./models"
torch.hub.set_dir(volume_path)


app = App(
    name="tacotron2",
    runtime=Runtime(
        cpu=1,
        memory="16Gi",
        gpu="T4",
        image=Image(
            python_version="python3.8",
            python_packages=[
                "numpy",
                "scipy",
                "pillow",
                "inflect",
                "librosa",
                "Unidecode",
                "torch",
                "inflect",
                "tqdm",
                "torchaudio",
                "speechbrain",
            ],
        ),
    ),
    volumes=[Volume(name="models", path=volume_path)],
)


def load_models():
    tacotron2 = torch.hub.load(
        "NVIDIA/DeepLearningExamples:torchhub", "nvidia_tacotron2", model_math="fp16"
    )
    tacotron2 = tacotron2.to("cuda")
    tacotron2.eval()

    waveglow = torch.hub.load(
        "NVIDIA/DeepLearningExamples:torchhub", "nvidia_waveglow", model_math="fp16"
    )
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow = waveglow.to("cuda")
    waveglow.eval()

    utils = torch.hub.load("NVIDIA/DeepLearningExamples:torchhub", "nvidia_tts_utils")

    return utils, tacotron2, waveglow


@app.rest_api(loader=load_models)
def to_speech(**inputs):
    """Convert text to audio using tacotron2.
    Args:
        text (str): text to turn to speech.

    Return:
        audio text
    """

    utils, tacotron2, waveglow = inputs["context"]

    sequences, lengths = utils.prepare_input_sequence([inputs["text"]])
    with torch.no_grad():
        mel, _, _ = tacotron2.infer(sequences, lengths)
        audio = waveglow.infer(mel)

    buffer_ = BytesIO()
    write(buffer_, 22050, audio[0].data.cpu().numpy())
    return {"audio": base64.b64encode(buffer_.getvalue()).decode("ascii")}
