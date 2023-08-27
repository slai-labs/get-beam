"""
### CodeLlama ###

** Pre-requisites **

Add your Huggingface API token to the Beam Secrets Manager, as `HUGGINGFACE_API_KEY`

** Run inference **

```sh
beam run app.py:generate -d '{"prompt": "YOUR PROMPT"}'
```
** Deploy API **

```sh
beam deploy app.py:generate
```
"""
from beam import App, Runtime, Image, Volume, VolumeType

import os
import torch
from transformers import LlamaForCausalLM, CodeLlamaTokenizer

base_model = "codellama/CodeLlama-7b-hf"

app = App(
    name="codellama",
    runtime=Runtime(
        cpu=1,
        memory="8Gi",
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
            commands=[
                "apt-get update && pip install git+https://github.com/huggingface/transformers"
            ],
        ),
    ),
    volumes=[
        Volume(
            name="model_weights",
            path="./model_weights",
            volume_type=VolumeType.Persistent,
        )
    ],
)


@app.rest_api()
def generate(**inputs):
    try:
        prompt = inputs["prompt"]
    except KeyError:
        prompt = '''
        def remove_non_ascii(s: str) -> str:
            \\"""<FILL_ME>
            return result
        '''

    tokenizer = CodeLlamaTokenizer.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        cache_dir="./model_weights",
        legacy=True,
        device_map={"": 0},
        use_auth_token=os.environ["HUGGINGFACE_API_KEY"],
    )

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        load_in_8bit=True,
        cache_dir="./model_weights",
        device_map={"": 0},
        use_auth_token=os.environ["HUGGINGFACE_API_KEY"],
    )

    with torch.no_grad():
        # Inference
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
