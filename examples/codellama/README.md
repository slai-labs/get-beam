# Serverless API for CodeLlama 

This example shows you how you can run CodeLlama on a serverless cloud GPU. It cold starts in ~10s.

## Pre-requisities 

First, create an account on [Beam](https://beam.cloud). It's free and you'll get 10 hours of GPU credit to start. Follow the steps in the onboarding to install the CLI and save the client credentials to your compute.

You'll also need to add your `HUGGINGFACE_API_KEY` to the [Beam Secret Manager](https://www.beam.cloud/dashboard/settings/secrets) to run this example.

![](./img/secret-list.png)

## Clone Inference Script

Once you've [installed the Beam CLI](https://docs.beam.cloud/getting-started/installation), download the inference script to your computer. You can clone this repo, or just run this command:

```sh
beam create-app codellama
```

## Deploy

When we're ready to deploy, we'll enter the shell and use the `beam deploy` command:

```sh
beam deploy app.py:generate
```

When you run this command, your browser window will open the Beam Dashboard. You can copy the cURL or Python request to call the API:

![](./img/call-api.png)
