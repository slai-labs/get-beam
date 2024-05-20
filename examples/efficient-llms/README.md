# Experimenting with LLMs on Small GPUs

In this article, we'll explain how to run LLM inference and fine-tuning on small (16Gi VRAM) GPUs.

Typically, LLMs require large GPUs, like A100s. But A100s are expensive. Luckily, it's possible to run productive LLM experiments on T4 GPUs by using various optimizations.

## Optimizing Model Resources with Unsloth

The optimization we're focused on is [Unsloth](https://github.com/unslothai/unsloth). Unsloth is a relatively new library that uses clever optimizations to substantially reduce the memory footprint of LLMs.

## Run Serverless LLaMA 3 in Under 20 Lines of Code

The magic of Unsloth lies in these lines:

```python
from unsloth import FastLanguageModel


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-Instruct-bnb-4bit"
)
```

### Deployment

You can deploy Llama 3 8B by downloading this app and running this command:

```
beam deploy app.py
```

### Calling the API

Once the app is deployed, you can call the API like this:

```sh
curl -X POST 'https://c4gxh-664bba8c2e84c8000a6bb6b9.apps.beam.cloud' \
-H 'Accept: */*' \
-H 'Accept-Encoding: gzip, deflate' \
-H 'Connection: keep-alive' \
-H 'Authorization: Basic [YOUR_AUTH_TOKEN]' \
-H 'Content-Type: application/json' \
-d '{"instruction":"you are a helpful AI assistant", "text":"help me book a dinner reservation"}'
```

If you look at the container logs, you'll see output like this:

```
2024-05-20T21:04:12.772453678Z:
INFO:     | Running task: 4345d785-b39e-4bec-98bc-5263625ef6b8
2024-05-20T21:04:45.294389413Z:
✅ ["<|begin_of_text|>Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nyou are a helpful AI assistant\n\n### Input:\nhelp me book a dinner reservation\n\n### Response:\nI'd be happy to help you book a dinner reservation. Can you please provide me with some details? What date and time would you like to make a reservation for? Are you looking to book a table for one person or a group? Also, do you have a specific restaurant in mind"]
⏰ Execution time: 8.40 seconds
2024-05-20T21:04:45.294421472Z:
INFO:     | Task complete: 4345d785-b39e-4bec-98bc-5263625ef6b8, duration: 32.45405387878418s
```

Importantly, this model is running on a T4 GPU with only 16Gi of VRAM.

### Next Steps

For adding more functionality, like fine-tuning, [this Colab notebook](https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing) is a great resource.
