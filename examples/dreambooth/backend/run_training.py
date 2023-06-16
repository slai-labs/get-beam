import pathlib
import requests
import subprocess
import hashlib
import os

"""
This function:
- takes a list of image URLs
- saves them to a Persistent Volume
- trains Dreambooth on the images
- saves them in a dedicated partition based on their user ID
"""

BASE_ROUTE = "./dreambooth"
pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"


def train_dreambooth(**inputs):

    user_id = inputs["user_id"]
    urls = inputs["image_urls"]
    instance_prompt = inputs["instance_prompt"]
    class_prompt = inputs["class_prompt"]

    # Create directories in Persistent Volume
    pathlib.Path(BASE_ROUTE).mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{BASE_ROUTE}/images/{user_id}").mkdir(parents=True, exist_ok=True)

    training_images_path = f"{BASE_ROUTE}/images/{user_id}"

    # Loop through the list of URLs provided and download each to a Persistent Volume
    for url in urls:
        response = requests.get(url)
        image_url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()

        if response.status_code == 200:
            with open(
                os.path.join(training_images_path, image_url_hash + ".png"), "wb"
            ) as f:
                f.write(response.content)
        else:
            print(f"Failed to save image from URL: {url}")

    # Dreambooth commands
    subprocess.run(
        [
            "python3.8",
            "-m",
            "accelerate.commands.accelerate_cli",
            "launch",
            f"--config_file=/workspace/default-config.yaml",
            "train_dreambooth.py",
            # Path to the pre-trained model
            f"--pretrained_model_name_or_path={pretrained_model_name_or_path}",
            # Path to the training data
            f"--instance_data_dir={training_images_path}",
            # Save trained model in the persistent volume, based on the user UUID
            f"--output_dir={BASE_ROUTE}/trained_models/{user_id}",
            "--prior_loss_weight=1.0",
            # Instance Prompt -- the specific instance of the image being fine-tuned, e.g. a [sks] man wearing sunglasses
            f"--instance_prompt={instance_prompt}",
            # Class Prompt -- the general category of the image being fine-tuned e.g. a man wearing sunglasses
            f"--class_prompt={class_prompt}",
            "--mixed_precision=no",
            "--resolution=512",
            "--train_batch_size=1",
            "--gradient_accumulation_steps=1",
            "--use_8bit_adam",
            "--gradient_checkpointing",
            "--set_grads_to_none",
            "--lr_scheduler=constant",
            "--lr_warmup_steps=0",
            # The two most useful levers in the training process
            # If the generated images don't match your prompt, you should consider increasing or decreasing the training steps and learning rate
            "--learning_rate=2e-6",
            "--max_train_steps=400",
        ],
        stdin=subprocess.PIPE,
        cwd="/workspace",
        env={**os.environ, "PYTHONPATH": "/workspace/__pypackages__:/workspace"},
    )


if __name__ == "__main__":
    user_id = "111111"
    instance_prompt = "a photo of a sks toy"
    class_prompt = "a photo of a toy"
    urls = [
        "https://huggingface.co/datasets/valhalla/images/resolve/main/2.jpeg",
        "https://huggingface.co/datasets/valhalla/images/resolve/main/3.jpeg",
        "https://huggingface.co/datasets/valhalla/images/resolve/main/5.jpeg",
        "https://huggingface.co/datasets/valhalla/images/resolve/main/6.jpeg",
    ]
    train_dreambooth(
        user_id=user_id,
        image_urls=urls,
        instance_prompt=instance_prompt,
        class_prompt=class_prompt,
    )