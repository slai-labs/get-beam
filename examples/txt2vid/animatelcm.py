"""
### Text to video using diffusers with AnimateLCM ###

**Deploy it as an API**

beam deploy modelscope.py:generate_video
"""
import torch
from diffusers import LCMScheduler, AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_gif

from beam import App, Output, Volume, Runtime, Image

cache_path = "./models"
step = 6

def load_models():
    device = "cuda"
    dtype = torch.float16

    repo = "wangfuyun/AnimateLCM"
    base = "emilianJR/epiCRealism"

    adapter = MotionAdapter.from_pretrained(
        repo, 
        torch_dtype=dtype,
    )
    pipe = AnimateDiffPipeline.from_pretrained(
        base, 
        motion_adapter=adapter, 
        torch_dtype=dtype,
        cache_dir=cache_path,
    ).to(device)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")

    pipe.enable_vae_slicing()
    pipe.enable_model_cpu_offload()

    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config, 
        timestep_spacing="trailing",
        beta_schedule="linear"
    )

    return pipe

app = App(
        name="t2v-animatelcm",
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
    outputs=[Output(path="animation.gif")],
    keep_warm_seconds=60,
)
def generate_video(**inputs):
    prompt = inputs["prompt"]
    pipe = inputs["context"]

    output = pipe(
        prompt=prompt, 
        negative_prompt="bad quality, worse quality, low resolution",
        num_frames=16,
        guidance_scale=2.0, 
        num_inference_steps=step,
    )
    export_to_gif(output.frames[0], "animation.gif")
