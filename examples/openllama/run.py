import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig

base_model = "openlm-research/open_llama_7b_preview_200bt"
subfolder = "open_llama_7b_preview_200bt_transformers_weights"


def generate_text(**inputs):

    prompt = inputs["prompt"]

    tokenizer = LlamaTokenizer.from_pretrained(
        base_model, subfolder=subfolder, cache_dir="./llama_weights"
    )
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        subfolder=subfolder,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir="./llama_weights",
    )

    tokenizer.bos_token_id = 1
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to("cuda")

    generation_config = GenerationConfig(
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
    )

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=128,
        )

    s = generation_output.sequences[0]
    decoded_output = tokenizer.decode(s, skip_special_tokens=True).strip()

    print(decoded_output)

    # Write text output to a text file, which we'll retrieve when the async task completes
    output_path = "output.txt"
    with open(output_path, "w") as f:
        f.write(decoded_output)


if __name__ == "__main__":
    prompt = "Simply put, the theory of relativity states that "
    generate_text(prompt=prompt)
