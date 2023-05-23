from transformers import AutoModelForCausalLM, AutoTokenizer

# Cached model
cache_path = "./santacoder"

# Huggingface model
model_id = "bigcode/santacoder"
device = "cuda"

def run(**inputs):
    # Takes prompt from webhook
    prompt = inputs["prompt"]

    # Tokenize and define model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device)
    model = model.to('cuda:0')
    
    # Generate output
    input = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated_text = model.generate(input, max_length= 60)
    generated_text = tokenizer.decode(generated_text[0])

    # Display and save output
    print(generated_text)
    with open("santacoder_output.txt", "w") as file:
        file.write(generated_text)
