# Pythia deployment with Beam

[Pythia-2.8b](https://huggingface.co/EleutherAI/pythia-2.8b) can be run on the cloud with a single command, using Beam.

1. [Create an account on Beam](https://beam.cloud). It's free and you don't need a credit card.

2. Install the Beam CLI:

```bash
curl https://raw.githubusercontent.com/slai-labs/get-beam/main/get-beam.sh -sSfL | sh
```

3. Clone this example to your computer:

```python
beam create-app pythia
```

4. Deploy and run inference:

```python
beam deploy app.py
```

This example is called through a task queue. Task queues are used for deploying
functions that run asynchronously on Beam. Here, the task queue takes a prompt
as one of its input fields. An example prompt and the real response from the
model are given below.

Example prompt:
> "Once upon a time"

Example response: 
> Once upon a time, the government had to make a decision.
