"""

*** Serverless Pygmalion API ***

> Run the API: beam serve app.py:predict
> Deploy it: beam deploy app.py:predict

"""

from beam import App, Runtime, Image, Volume, RequestLatencyAutoscaler
from transformers import AutoTokenizer, AutoModelForCausalLM

# Beam volume to store cached models
CACHE_PATH = "./cached_models"

app = App(
    name="pygmalion",
    runtime=Runtime(
        cpu=2,
        memory="32Gi",
        gpu="A10G",
        image=Image(
            python_version="python3.9",
            python_packages=[
                "transformers",
                "accelerate",
                "torch",
                "bitsandbytes",
                "scipy",
                "protobuf",
            ],  # You can also add a path to a requirements.txt instead
        ),
    ),
    # Storage volume for cached models
    volumes=[Volume(name="cached_models", path=CACHE_PATH)],
)


# Pre-load models: this function runs once when the container boots
def load_models():
    tokenizer = AutoTokenizer.from_pretrained(
        "PygmalionAI/pygmalion-6b", cache_dir=CACHE_PATH
    )
    model = AutoModelForCausalLM.from_pretrained(
        "PygmalionAI/pygmalion-6b",
        load_in_8bit=True,
        device_map="auto",
        cache_dir=CACHE_PATH,
    )
    return model, tokenizer


# Autoscale by request latency - will spin up add'l replicas if latency exceeds 30s
autoscaler = RequestLatencyAutoscaler(desired_latency=30, max_replicas=5)


# Rest API initialized with loader and autoscaler
@app.rest_api(loader=load_models, autoscaler=autoscaler)
def predict(**inputs):
    # Retrieve cached model from loader
    model, tokenizer = inputs["context"]
    # Input from API request
    prompt = inputs["prompt"]

    # Inference
    inputs = tokenizer(prompt, return_tensors="pt")
    generate_ids = model.generate(inputs.input_ids.to("cuda"), max_length=30)
    result = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    print(result)

    return {"prediction": result}
