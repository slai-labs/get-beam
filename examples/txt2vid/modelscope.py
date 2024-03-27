"""
### Text to video using diffusers with modelscope ###

**Deploy it as an API**

beam deploy modelscope.py:generate_video
"""
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video

from beam import App, Output, Volume, Runtime, Image

cache_path = "./models"

def load_models():
    device = "cuda"
    dtype = torch.float16

    pipe = DiffusionPipeline.from_pretrained(
        "damo-vilab/text-to-video-ms-1.7b", 
        torch_dtype=dtype,
        cache_dir=cache_path,
    ).to(device)

    pipe.enable_vae_slicing()
    pipe.enable_model_cpu_offload()

    return pipe

app = App(
        name="t2v-modelscope",
        runtime=Runtime(
            cpu=2,
            memory="16Gi",
            gpu="A10G",
            image=Image(
                python_version="python3.8",
                python_packages="requirements.txt",
            ),
        ),
        volumes=[
            Volume(name="models", path="./models"),
        ],
    )

@app.task_queue(
    loader=load_models,
    outputs=[Output(path="modelscopet2v.mp4")],
    keep_warm_seconds=60,
)
def generate_video(**inputs):
    prompt = inputs["prompt"]
    pipe = inputs["context"]

    output = pipe(
        prompt=prompt, 
        negative_prompt="bad quality, worse quality, low resolution",
    )
    export_to_video(output.frames[0], "modelscopet2v.mp4", fps=10)
