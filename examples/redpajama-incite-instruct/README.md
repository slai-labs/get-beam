# RedPajama-INCITE deployment with Beam

[RedPajama-INCITE](https://huggingface.co/togethercomputer/RedPajama-INCITE-Instruct-3B-v1) can be deployed on Beam for development
on serverless GPUs.

1. [Create an account on Beam](https://beam.cloud). It's free and you don't need a credit card.

2. Install the Beam CLI:

```bash
curl https://raw.githubusercontent.com/slai-labs/get-beam/main/get-beam.sh -sSfL | sh
```

3. Clone this example to your computer:

```python
beam create-app redpajama-incite-instruct
```

4. Deploy and run inference:

```python
beam deploy app.py
```

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
