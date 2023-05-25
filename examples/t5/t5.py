from transformers import T5Tokenizer, T5ForConditionalGeneration

# Cached model
cache_path = "./t5"

# Huggingface model
model_id = "t5-large"

def run(**inputs):
    # Takes prompt from webhook
    prompt = inputs["prompt"]

    # Tokenize and define model
    tokenizer = T5Tokenizer.from_pretrained(model_id)
    model = T5ForConditionalGeneration.from_pretrained(model_id, cache_dir=cache_path)
    
    # Generate output
    input = tokenizer(prompt, return_tensors="pt", padding=True)
    generate_text = model.generate(
        input_ids=input["input_ids"],
        attention_mask=input["attention_mask"],
        do_sample=False,)
    generated_text = tokenizer.batch_decode(generate_text, skip_special_tokens=True)
    generated_text = generated_text[0]

    # Display and save output
    print(generated_text)
    with open("t5_output.txt", "w") as file:
        file.write(generated_text)
