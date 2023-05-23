from transformers import pipeline, AutoTokenizer

# Cached model
cache_path = "./cerebras-gpt"

# Huggingface model
model_id = "cerebras/Cerebras-GPT-1.3B"

def run(**inputs):
    # Takes prompt from webhook
    prompt = inputs["prompt"]

    # Tokenize and define model
    tokenizer = AutoTokenizer.from_pretrained("cerebras/Cerebras-GPT-1.3B")
    pipe = pipeline(
        "text-generation",
        model=model_id,
        tokenizer=tokenizer)
    
    # Generate output
    generated_text = pipe(
        prompt,
        max_length=50,
        do_sample=False,
        no_repeat_ngram_size=2)[0]
    
    # Display and save output
    print(generated_text['generated_text'])
    with open("cerebrasgpt_output.txt", "w") as file:
        file.write(generated_text['generated_text'])
