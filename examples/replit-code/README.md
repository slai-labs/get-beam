# Replit Code deployment with Beam

[Replit Code](https://huggingface.co/replit/replit-code-v1-3b) can be run on the cloud with a single command, using Beam.

1. [Create an account on Beam](https://beam.cloud). It's free and you don't need a credit card.

2. Install the Beam CLI:

```bash
curl https://raw.githubusercontent.com/slai-labs/get-beam/main/get-beam.sh -sSfL | sh
```

3. Clone this example to your computer:

```python
beam create-app replit-code
```

4. Deploy and run inference:

```python
beam deploy app.py
```

This example is called through a task queue. Task queue are used for deploying
functions that run asynchronously on Beam. Here, the task queue takes a prompt
as one of its input fields. An example prompt and the real response from the
model are given below.

Example prompt:
> "def fibonacci(n): "

Example response: 
> def fibonacci(n): # n is the number of terms to return
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)
  print(fibonacci(10))
