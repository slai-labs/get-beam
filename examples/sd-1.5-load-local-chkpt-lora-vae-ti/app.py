"""
### Load local weights for base model Stable Diffusion 1.5 on Beam ###

**Deploy it as an API**

beam deploy app.py:generate_image
"""
import base64
import os
import torch

from beam import App, Runtime, Image, Output, Volume, VolumeType
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, AutoencoderKL


# The volume storing the models(shared)
volume_name = "models"
volume_path = "./models" # add your own path from your Beam Volume

# Image name used to save the generated image
img_name = "output_img.png"

# model checkpoint path(add your own path)
model_path = f"{volume_path}/Anything-V3.0-pruned-fp32.safetensors"
# lora path(add your own lora name that you have on your Beam Volume)
lora_path = f"{volume_path}/"
lora_name = "Crayon.safetensors" # this is actually the name of the file on Beam Volume
# textual inversion path(add your own path)
ti_path = f"{volume_path}/1vanakn0ll.pt"
# vae path(add your own path)
vae_path = f"{volume_path}/vae-ft-mse-840000-ema-pruned.safetensors"

# model settings(params)
image_width = 512
image_height = 768
negative_prompt = "(ugly:1.3), (fused fingers), (too many fingers), (bad anatomy:1.2), (watermark:1.2), (words), letters, untracked eyes, asymmetric eyes"
sampler = DPMSolverMultistepScheduler.from_config # = 'DPM++ 2M' - scheduler - set bellow after the pipe is created
sampler_type = "use_karras_sigmas" # = 'Karras' - scheduler type - set bellow after the pipe is created
sampling_steps = 20
cfg_scale = 7.5 # CFG scale
model_seed = 1736616725
clip_skip = 2
# Precision Consistency - Ensured that torch_dtype=torch.float32 is consistently used
# when loading models and other components.
precision_consistency_val = "32" # or 16 - to fix the "Input type (c10::Half) and bias type (float) mismatch" error
                                 # also "RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu"

# The environment your app runs on
app = App(
    name="sd-1.5-load-local-chkpt-lora-vae-ti",
    runtime=Runtime(
        cpu=4,
        memory="16Gi",
        gpu="T4",
        image=Image(
            python_version="python3.10",
            python_packages="requirements.txt",
            # Shell commands that run when the container first starts
            # This is used to install cuda library to fix the "OSError: libcudart.so.11.0: cannot open shared object file: No such file or directory"
            commands=[
                "apt-get update && apt-get install wget"
                " && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb",
                "dpkg -i cuda-keyring_1.1-1_all.deb",
                "lsb_release -a && apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg libpq-dev cuda=11.8.0-1 libcudnn8=8.9.2.*-1+cuda11.8",
            ],
        ),
    ),
    volumes=[
        Volume(
            name=volume_name,
            path=volume_path,
            volume_type=VolumeType.Shared,
        )
    ],
)

# check if GPU or CPU is available - this is available on the Beam platform
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# check if MPS is available on Apple M1 chips if running locally on a MAC
# MPS is not available on the Beam platform
try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

# print(f"Device available for model: {device}")

# Deploy as a REST API
@app.rest_api()

# Deploy as an async task queue - for big images
# since the REST API has a 60s timeout
# @app.task_queue(
#     loader=load_models,
#     outputs=[Output(path="output.png")],
#     keep_warm_seconds=60,
# )

# main function that runs the inference
def generate_image(**inputs):
    # Grab inputs passed to the API
    try:
        prompt = inputs["prompt"]
    # Use a default prompt if none is provided
    except KeyError:
        prompt = "a renaissance style photo of elon musk"

    # print(f"Image prompt: {prompt}")

    # Set the device to run the model on
    # print(f"Device available for model: {device}")
    # print(f"MPS available for model: {torch.backends.mps.is_available()}")
    # print(f"Cuda available for model: {torch.cuda.is_available()}")
    # print(f"CUDA version: {torch.version.cuda}")

    # Precision Consistency - Ensured that torch_dtype=torch.float32 is consistently used
    # precision_consistency = torch["float" + precision_consistency_val]
    precision_consistency = getattr(torch, "float" + precision_consistency_val)

    # Special torch method to improve performance
    torch.backends.cuda.matmul.allow_tf32 = True

    # Load the model checkpoint
    # `StableDiffusionPipeline` is a class that represents a pipeline for stable diffusion, a
    # technique used in image generation tasks. It is used to load a pre-trained model
    # checkpoint, run inference on the model, and generate images based on a given prompt.
    # The pipeline handles the processing of the input prompt, running the inference steps on the
    # model, and producing the final image output.
    pipe = StableDiffusionPipeline.from_single_file(
        # Run inference on the specific model trained(checkpoint) from the volume
        model_path,
        torch_dtype=precision_consistency,
        variant="fp32",
        # The `cache_dir` arg is used to cache the model in between requests
        cache_dir=volume_path,
        safety_checker=None,
        use_safetensors=True,
        device_map="auto"
    ).to("cuda")
    # It also includes functionalities for memory-efficient attention mechanisms and optimization for image generation tasks.
    pipe.enable_xformers_memory_efficient_attention()

    # VAE - Load the VAE with float32 precision
    pipe.vae = AutoencoderKL.from_single_file(vae_path, torch_dtype=precision_consistency).to("cuda")

    # LORAs - Load LORAs with float32 precision
    pipe.load_lora_weights(lora_path, weight_name=lora_name, torch_dtype=precision_consistency)

    # Textual Inversion - Load Textual Inversion with float32 precision
    pipe.load_textual_inversion(ti_path, torch_dtype=precision_consistency)

    # PIPELINE SETTINGS
    # set schedule type
    sch_config = pipe.scheduler.config
    sch_config[sampler_type] = True
    # print(f"schedule config: {sch_config}")
    
    # set the scheduler
    pipe.scheduler = sampler(sch_config)
    # print(f"sampler(scheduler) info: {pipe.scheduler}")
    # print(f"sampler(scheduler) use_karras_sigmas: {pipe.scheduler.use_karras_sigmas}")

    # Image generation
    with torch.inference_mode():
        with torch.autocast("cuda", dtype=precision_consistency):
            image = pipe(prompt, width=image_width, height=image_height,
                         negative_prompt=negative_prompt,
                         num_inference_steps=sampling_steps,
                         guidance_scale=cfg_scale,
                         generator = torch.Generator(device="cuda").manual_seed(model_seed),
                         clip_skip=clip_skip,
                        ).images[0]

    print(f"Saved Image: {image}")

    # Save the generated image to the volume
    image.save(img_name)

    # decode the image to base64
    img_path = os.path.abspath(img_name);
    img_base64=base64.b64encode(open(img_path, "rb").read()).decode("UTF-8")

    # print it to the console
    # print(f"Base64 Image: {img_base64}")
    
    # return the base64 image
    return {"data": img_base64}