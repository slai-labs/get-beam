"""Main beam handler to do tacotron2 inference"""

import os
import pathlib
from io import BytesIO
import numpy as np
import urllib
import base64
from tqdm import tqdm
import hashlib

import torch
import numpy as np
from text import text_to_sequence
from models import Tacotron2
from hparams import hparams as hps
from utils import mode, to_arr
from audio import save_wav, inv_melspectrogram
from typing import Union
from scipy.io.wavfile import write

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False

device = "cuda"
url = "https://github.com/BogiHsu/Tacotron2-PyTorch/releases/download/lj-200k-b512/ckpt_200000"
dst_dir = '/volumes/vol/'
target_filename = 'tacotron.ckpt'
target_path = os.path.join(dst_dir, target_filename)
torch.hub.set_dir(dst_dir)

tacotron2 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tacotron2', model_math='fp16')
tacotron2 = tacotron2.to('cuda')
tacotron2.eval()

waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp16')
waveglow = waveglow.remove_weightnorm(waveglow)
waveglow = waveglow.to('cuda')
waveglow.eval()

utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tts_utils')

def to_speech(text: str) -> str:
    """Convert text to audio using tacotron2.
    Args:
        text (str): text to turn to speech.

    Return:
        audio text
    """

    sequences, lengths = utils.prepare_input_sequence([text])
    with torch.no_grad():
        mel, _, _ = tacotron2.infer(sequences, lengths)
        audio = waveglow.infer(mel)

    buffer_ = BytesIO()
    write(buffer_, 22050, audio[0].data.cpu().numpy())
    return {
        "audio": base64.b64encode(buffer_.getvalue()).decode("ascii")
    }
