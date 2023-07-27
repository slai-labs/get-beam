from beam import App, Runtime, Image, Output, Volume

from transformers import AutoModelForCausalLM, AutoTokenizer

# Cached model
cache_path = "./santacoder"

# Huggingface model
model_id = "bigcode/santacoder"
device = "cuda"


# The environment your code will run on
app = App(
    name="santacoder",
    runtime=Runtime(
        cpu=8,
        memory="32Gi",
        gpu="A10G",
        image=Image(
            python_version="python3.8",
            python_packages=[
                "diffusers[torch]>=0.10",
                "transformers",
                "torch",
                "pillow",
                "accelerate",
                "safetensors",
                "xformers",
            ],
        ),
    ),
    volumes=[Volume(path="./santacoder", name="santacoder")],
)


@app.task_queue(outputs=[Output(path="santacoder_output.txt")])
def run(**inputs):
    # Takes prompt from webhook
    prompt = inputs["prompt"]

    # Tokenize and define model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, cache_dir=cache_path
    ).to(device)
    model = model.to("cuda:0")

    # Generate output
    input = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated_text = model.generate(input, max_length=60)
    generated_text = tokenizer.decode(generated_text[0])

    # Display and save output
    print(generated_text)
    with open("santacoder_output.txt", "w") as file:
        file.write(generated_text)
