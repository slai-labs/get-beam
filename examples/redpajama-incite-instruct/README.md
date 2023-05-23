# RedPajama-INCITE-Instruct deployment with Beam

[DLite](https://huggingface.co/togethercomputer/RedPajama-INCITE-Instruct-3B-v1) can be deployed on Beam for development
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

## Deploying RedPajama-INCITE-Instruct-3B-v1

Create a local copy of the project by running:

```beam create-app redpajama-incite-instruct```

And deploy it using:

```beam deploy app.py```

This example is called through a webhook. Webhooks are used for deploying
functions that run asynchronously on Beam. Here, the webhook takes a prompt
as one of its input fields. An example prompt and the real response from the
model are given below.

Example prompt:
> The capital of France is

Example response: 
>  of Environmental Conservation to Provide $3.5 Million to Fund Restoration of New York’s High Peaks
New York State Department of Environmental Conservation to Provide $3.5 Million to Fund Restoration of New York’s High Peaks
The New York State Department of Environmental Conservation (DEC) announced today that it will provide $3.5 million to fund the restoration of the High Peaks region of the Adirondack Park. The funding is part of the $30 million that DEC has committed to restore and improve the Adirondack Park’s High
