from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Cached model
cache_path = "./codet5p"

# Huggingface model
model_id = "Salesforce/codet5p-6b"
device = "cuda"

def run(**inputs):
    # Takes prompt from task queue
    prompt = inputs["prompt"]

    # Tokenize and define model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        cache_dir=cache_path).to(device)

    # Generate output
    encoding = tokenizer(
        prompt,
        return_tensors="pt").to(device)
    encoding['decoder_input_ids'] = encoding['input_ids'].clone()
    outputs = model.generate(**encoding, max_length=250)
    generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Display and save output
    print(generated_code)
    with open("codet5p_output.txt", "w") as file:
        file.write(generated_code)
