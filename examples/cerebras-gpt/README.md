# Cerebras-GPT deployment with Beam

[Cerebras-GPT](https://huggingface.co/cerebras/Cerebras-GPT-1.3B) can be deployed on Beam for development
on serverless GPUs.

## Getting started with Beam

### Create an Account
- [Create an account](https://www.beam.cloud/)
- Grab your API keys from the [dashboard](https://www.beam.cloud/dashboard/settings/api-keys).

### Install Beam CLI
In your terminal, run:

```curl https://raw.githubusercontent.com/slai-labs/get-beam/main/get-beam.sh -sSfL | sh```

### Register API Keys
Run this in your terminal - you’ll be prompted to paste in your API keys:

```beam configure```

### Install Beam SDK
Finally, install the Beam SDK:

```pip install beam-sdk```

## Deploying Cerebras-GPT-1.3B

Create a local copy of the project by running:

```beam create-app cerebras-gpt```

And deploy it using:

```beam deploy app.py```

This example is called through a webhook. Webhooks are used for deploying
functions that run asynchronously on Beam. Here, the webhook takes a prompt
as one of its input fields. An example prompt and the real response from the
model are given below.

Example prompt: 
> Once upon a time

Example response: 
> Once upon a time, the world was a place of peace and prosperity. But the people of the land were not content with peace. They wanted to be free. The people were divided into two groups. The first group was the
