from beam import App, Runtime, Image, Output, Volume

from transformers import GPTNeoXForCausalLM, AutoTokenizer

# Cached model
cache_path = "./pythia"

# Huggingface model
model_id = "EleutherAI/pythia-2.8b"


# The environment your code will run on
app = App(
    name="pythia",
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
    volumes=[Volume(path="./pythia", name="pythia")],
)


@app.run(outputs=[Output(path="pythia_output.txt")])
def inference(**inputs):
    # Takes prompt from webhook
    prompt = inputs["prompt"]

    # Tokenize and define model
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision="step3000", cache_dir=cache_path
    )
    model = GPTNeoXForCausalLM.from_pretrained(
        model_id, revision="step3000", cache_dir=cache_path
    )

    # Generate output
    input = tokenizer(prompt, return_tensors="pt")
    generate_text = model.generate(**input)
    generated_text = tokenizer.decode(generate_text[0])

    # Display and save output
    print(generated_text)
    with open("pythia_output.txt", "w") as file:
        file.write(generated_text)

