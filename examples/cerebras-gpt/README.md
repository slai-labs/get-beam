# Cerebras-GPT deployment with Beam

[Cerebras-GPT](https://huggingface.co/cerebras/Cerebras-GPT-1.3B) can be deployed on Beam for development
on serverless GPUs.

## Deploying Cerebras-GPT-1.3B

1. [Create an account on Beam](https://beam.cloud). It's free and you don't need a credit card.

2. Install the Beam CLI:

```bash
curl https://raw.githubusercontent.com/slai-labs/get-beam/main/get-beam.sh -sSfL | sh
```

3. Clone this example to your computer:

```python
beam create-app cerebras-gpt
```

4. Deploy and run inference:

```python
beam deploy app.py
```

### Install Beam SDK
Finally, install the Beam SDK:

```pip install beam-sdk```

This example is called through a webhook. Webhooks are used for deploying
functions that run asynchronously on Beam. Here, the webhook takes a prompt
as one of its input fields. An example prompt and the real response from the
model are given below.

Example prompt: 
> Once upon a time

Example response: 
> Once upon a time, the world was a place of peace and prosperity. But the people of the land were not content with peace. They wanted to be free. The people were divided into two groups. The first group was the
