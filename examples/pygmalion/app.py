from beam import App, Runtime, Image, Volume, RequestLatencyAutoscaler
from transformers import AutoTokenizer, AutoModelForCausalLM

# Beam Volume to store cached models
CACHE_PATH = "./cached_models"

app = App(
    name="pygmalian",
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
    # Storage Volume for model weights
    volumes=[Volume(name="cached_models", path=CACHE_PATH)],
)

# Autoscale by request latency
autoscaler = RequestLatencyAutoscaler(desired_latency=1, max_replicas=5)


# This function runs once when the container boots
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


# Rest API initialized with loader and autoscaler
@app.rest_api(loader=load_models, autoscaler=autoscaler)
def predict(**inputs):
    # Retrieve cached model from loader
    model, tokenizer = inputs["context"]

    prompt = inputs["prompt"]
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate
    generate_ids = model.generate(inputs.input_ids.to('cuda'), max_length=30)
    result = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    print(result)

    return {"prediction": result}