import torch
from peft import PeftModel
from transformers import GenerationConfig, LlamaTokenizer

from utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def call_model(
    prompter,
    tokenizer,
    model,
    input,
    do_sample=True,
    temperature=0.3,
    top_p=0.75,
    top_k=40,
    num_beams=1,
    max_new_tokens=600,
    **kwargs,
):
    prompter: Prompter = prompter
    tokenizer: LlamaTokenizer = tokenizer
    model: PeftModel = model

    prompt = prompter.generate_prompt(input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        do_sample=do_sample,
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

    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    prompt_response = prompter.get_response(output)

    print(prompt_response)
    return {"prompt_response": prompt_response}
