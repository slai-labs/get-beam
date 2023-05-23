# Santacoder deployment with Beam

[Santacoder](https://huggingface.co/bigcode/santacoder) can be deployed on Beam for development
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

## Deploying Santacoder

Create a local copy of the project by running:

```beam create-app santacoder```

And deploy it using:

```beam deploy app.py```

This example is called through a webhook. Webhooks are used for deploying
functions that run asynchronously on Beam. Here, the webhook takes a prompt
as one of its input fields. An example prompt and the real response from the
model are given below.

Example prompt:
> "def iterative_count_to_10():"

Example response: 
> def iterative_count_to_10():
    """
    >>> iterative_count_to_10()
    10
    """
    return 10
    def iterative_count_to_100():
    """
    >>> iterative_count_to_
