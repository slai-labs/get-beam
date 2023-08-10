import os
import json
import sys

from beam import App, Runtime, Image, Volume
from finetune import train

import fire
import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter

BASE_MODEL = "decapoda-research/llama-7b-hf"

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
    volumes=[
        Volume(name="lora-alpaca", path="./lora-alpaca"),
        Volume(name="pretrained-models", path="./pretrained-models"),
    ],
)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def loader():
    load_8bit: bool = True
    base_model: str = BASE_MODEL
    lora_weights: str = "./lora-alpaca"
    prompt_template: str = ""  # The prompt template to use, will default to alpaca.
    base_model = base_model or os.environ.get("BASE_MODEL", "")

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=f"./pretrained-models/{base_model}",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            low_cpu_mem_usage=True,
            cache_dir=f"./pretrained-models/{base_model}",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    return {
        "prompter": prompter,
        "tokenizer": tokenizer,
        "model": model,
    }


@app.schedule(when="every 24h")
def train_model():
    train(base_model=BASE_MODEL)


@app.rest_api(loader=loader)
def evaluate(
    instruction,
    context={},
    input=None,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    max_new_tokens=128,
    **kwargs,
):
    prompter: Prompter = context.get("prompter")
    tokenizer: LlamaTokenizer = context.get("tokenizer")
    model: PeftModel = context.get("model")

    prompt = prompter.generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )

    # Without streaming
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    print(generation_output)
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    print(json.dumps(output))
    print({"verify": "stuff works"})
    prompt_response = prompter.get_response(output)
    print(prompt_response)
    return prompt_response


if __name__ == "__main__":
    train_model()
