from beam import App, Runtime, Image
import time
from unsloth import FastLanguageModel


alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

app = App(
    name="llama3-inference",
    runtime=Runtime(
        cpu=4,
        memory="16Gi",
        gpu="T4",
        image=Image(
            python_version="python3.10",
            python_packages=[
                "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git",
                "xformers<0.0.26",
                "trl",
                "peft",
                "accelerate",
                "bitsandbytes",
            ],
        ),
    ),
)


def preload():
    max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
    dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/llama-3-8b-Instruct-bnb-4bit",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )
    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

    return model, tokenizer


@app.rest_api(loader=preload)
def predict(**inputs):
    # Measure execution time
    start_time = time.time()

    # Retrieve cached model and tokenizer from loader function
    model, tokenizer = inputs["context"]

    # Format prompt
    inputs = tokenizer(
        [
            alpaca_prompt.format(
                inputs["instruction"],  # instruction
                inputs["text"],  # input
                "",  # output - leave this blank for generation!
            )
        ],
        return_tensors="pt",
    ).to("cuda")

    # Inference
    outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True)
    result = tokenizer.batch_decode(outputs)

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"✅ {result}")
    print(f"⏰ Execution time: {execution_time:.2f} seconds")

    return {"generation": result}
