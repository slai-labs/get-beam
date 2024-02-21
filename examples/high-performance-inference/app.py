from beam import App, Runtime, Image, Volume, RequestLatencyAutoscaler
from transformers import AutoModelForCausalLM, AutoTokenizer

# Beam Volume to store cached models
CACHE_PATH = "./cached_models"

device = "cuda"

app = App(
    name="inference-quickstart",
    runtime=Runtime(
        cpu=1,
        memory="8Gi",
        gpu="A10G",
        image=Image(
            python_version="python3.8",
            python_packages=[
                "transformers",
                "torch",
            ],  # You can also add a path to a requirements.txt instead
        ),
    ),
    # Storage Volume for model weights
    volumes=[Volume(name="cached_models", path=CACHE_PATH)],
)

# Autoscale by request latency
autoscaler = RequestLatencyAutoscaler(desired_latency=30, max_replicas=3)


# This function runs once when the container boots
def load_models():
    model = AutoModelForCausalLM.from_pretrained(
        "roneneldan/TinyStories-1M", cache_dir=CACHE_PATH
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/gpt-neo-125M", cache_dir=CACHE_PATH
    )

    return model, tokenizer


# Rest API initialized with loader and autoscaler
@app.rest_api(loader=load_models, autoscaler=autoscaler)
def predict(**inputs):
    # Retrieve cached model from loader
    model, tokenizer = inputs["context"]
    # Grab the prompt from the API
    try:
        prompt = inputs["prompt"]
    # Use a default prompt if none is provided
    except KeyError:
        prompt = "Once upon a time there was"

    # Generate
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    output = model.generate(
        inputs, max_length=1000, num_beams=1, pad_token_id=tokenizer.eos_token_id
    )
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)

    print(output_text)

    return {"prediction": output_text}
