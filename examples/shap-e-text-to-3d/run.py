# This code has been adapted from the shap-e repository's example ipynb files
import torch

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import (
    create_pan_cameras,
    decode_latent_images,
    decode_latent_mesh,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 4
guidance_scale = 15.0
render_mode = "nerf"  # you can change this to 'stf'
size = 64  # this is the size of the renders; higher values take longer to render.


def load_models():
    xm = load_model("transmitter", device=device)
    model = load_model("text300M", device=device)
    diffusion = diffusion_from_config(load_config("diffusion"))

    return dict(xm=xm, model=model, diffusion=diffusion)


def generate_model(prompt: str, **inputs):
    models: dict = inputs["context"]

    xm = models["xm"]
    model = models["model"]
    diffusion = models["diffusion"]

    latents = sample_latents(
        batch_size=batch_size,
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=dict(texts=[prompt] * batch_size),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=64,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )

    # Get only the first latent for this example
    latent = latents[0]
    cameras = create_pan_cameras(size, device)

    images = decode_latent_images(xm, latent, cameras, rendering_mode=render_mode)
    # Save the image to the output file
    with open("output.png", "wb") as f:
        images[0].save(f, format="png")

    # Save the mesh to the output file
    with open(f"model.ply", "wb") as f:
        decode_latent_mesh(xm, latent).tri_mesh().write_ply(f)

    print(f"Saved output to output.png and model.ply")
