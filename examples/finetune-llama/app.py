from math import ceil

from beam import App, Runtime, Image, Volume
from helpers import get_newest_checkpoint, base_model
from training import train, load_models
from datasets import load_dataset
from inference import call_model


# The environment your code runs on 
app = App(
    "llama-lora",
    runtime=Runtime(
        cpu=4,
        memory="32Gi",
        gpu="A10G",
        image=Image(
            python_version="python3.10",
            python_packages="requirements.txt",
        ),
    ),
    # Mount Volumes for fine-tuned models and cached model weights
    volumes=[
        Volume(name="checkpoints", path="./checkpoints"),
        Volume(name="pretrained-models", path="./pretrained-models"),
    ],
)


# Fine-tuning
@app.run()
def train_model():
    # Trained models will be saved to this path
    beam_volume_path = "./checkpoints"

    # Load dataset -- for this example, we'll use the vicgalle/alpaca-gpt4 dataset hosted on Huggingface:
    # https://huggingface.co/datasets/vicgalle/alpaca-gpt4
    dataset = load_dataset("vicgalle/alpaca-gpt4")
    
    # Adjust the training loop based on the size of the dataset
    samples = len(dataset["train"])
    val_set_size = ceil(0.1 * samples)

    train(
        base_model=base_model,
        val_set_size=val_set_size,
        data=dataset,
        output_dir=beam_volume_path,
    )


# ---------------------------------------------------------------------------- #
#                                Inference API                                 #
# ---------------------------------------------------------------------------- #
@app.rest_api()
def run_inference(**inputs):
    # Inputs passed to the API
    input = inputs["input"]

    # Grab the latest checkpoint
    checkpoint = get_newest_checkpoint()
    
    # Initialize models with latest fine-tuned checkpoint
    models = load_models(checkpoint=checkpoint)

    model = models["model"]
    tokenizer = models["tokenizer"]
    prompter = models["prompter"]

    # Generate text response
    response = call_model(
        input=input, model=model, tokenizer=tokenizer, prompter=prompter
    )
    return response