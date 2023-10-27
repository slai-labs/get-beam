"""
### CodeLlama ###

** Pre-requisites **

Add your Huggingface API token to the Beam Secrets Manager, as `HUGGINGFACE_API_KEY`

** Deploy API **

```sh
beam deploy app.py:generate
```
"""
from beam import App, Runtime, Image, Volume, VolumeType

import os
import torch
from transformers import LlamaForCausalLM, CodeLlamaTokenizer

# CodeLlama 7B
base_model = "codellama/CodeLlama-7b-hf"

# Beam Volume Path to cache model weights
cache_path = "model_weights"

# This is the compute environment your code will run on
app = App(
    name="codellama",
    runtime=Runtime(
        cpu=4,
        memory="16Gi",
        gpu="T4",
        image=Image(
            python_packages=[
                "accelerate",
                "bitsandbytes",
                "scipy",
                "protobuf",
                "accelerate",
                "torch",
                "sentencepiece",
            ],
            # Shell commands that run when the container first starts
            commands=[
                "apt-get update && pip install git+https://github.com/huggingface/transformers"
            ],
        ),
    ),
    # Mount a storage volume to cache the model weights
    volumes=[
        Volume(
            name="model_weights",
            path=cache_path,
            volume_type=VolumeType.Persistent,
        )
    ],
)


# Pre-load the models for inference
def load_models():
    tokenizer = CodeLlamaTokenizer.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        cache_dir=cache_path,
        legacy=True,
        device_map={"": 0},
    )

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        load_in_8bit=True,
        cache_dir=cache_path,
        device_map={"": 0},
    )

    return tokenizer, model


# This decorator allows us to deploy the API on Beam
@app.rest_api(loader=load_models)
def generate(**inputs):
    # Grab the prompt from the API
    try:
        prompt = inputs["prompt"]
    # Use a default prompt if none is provided
    except KeyError:
        prompt = '''
        def remove_non_ascii(s: str) -> str:
            \\"""<FILL_ME>
            return result
        '''

    # Retrieve cached model and tokenizer from loader
    tokenizer, model = inputs["context"]

    # Inference
    with torch.no_grad():
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to("cuda")
        generated_ids = model.generate(
            input_ids, max_new_tokens=128, pad_token_id=tokenizer.eos_token_id
        )

        filling = tokenizer.batch_decode(
            generated_ids[:, input_ids.shape[1] :], skip_special_tokens=True
        )[0]

    response = prompt.replace("<FILL_ME>", filling)
    print(response)

    return {"response": response}
