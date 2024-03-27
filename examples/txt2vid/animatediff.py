"""
### Text to video using diffusers with animatediff ###

**Deploy it as an API**

beam deploy animatediff.py:generate_video
"""
import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
from diffusers.utils import export_to_gif

from beam import App, Output, Volume, Runtime, Image

cache_path = "./models"

def load_models():
    device = "cuda"
    dtype = torch.float16

    repo = "guoyww/animatediff-motion-adapter-v1-5-2"
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
    scheduler = DDIMScheduler.from_pretrained(
        base,
        subfolder="scheduler",
        clip_sample=False,
        timestep_spacing="linspace",
        beta_schedule="linear",
        steps_offset=1,
    )
    pipe.scheduler = scheduler

    pipe.enable_vae_slicing()
    pipe.enable_model_cpu_offload()

    return pipe

app = App(
        name="t2v-animatediff",
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
        guidance_scale=7.5,
        num_inference_steps=50,
        generator=torch.Generator("cpu").manual_seed(49),
    )
    export_to_gif(output.frames[0], "animation.gif")
