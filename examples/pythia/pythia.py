from transformers import GPTNeoXForCausalLM, AutoTokenizer

# Cached model
cache_path = "./pythia"

# Huggingface model
model_id = "EleutherAI/pythia-2.8b"

def run(**inputs):
    # Takes prompt from webhook
    prompt = inputs["prompt"]

    # Tokenize and define model
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision="step3000",
        cache_dir=cache_path)
    model = GPTNeoXForCausalLM.from_pretrained(
        model_id,
        revision="step3000",
        max_new_tokens="50",
        cache_dir=cache_path)
    
    # Generate output
    input = tokenizer(prompt, return_tensors="pt")
    generate_text = model.generate(**input)
    generated_text = tokenizer.decode(generate_text[0])

    # Display and save output
    print(generated_text)
    with open("pythia_output.txt", "w") as file:
        file.write(generated_text)
