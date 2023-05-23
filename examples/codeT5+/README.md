# CodeT5+ deployment with Beam

[CodeT5+](https://github.com/salesforce/CodeT5/tree/main/CodeT5+) can be deployed on Beam for development
on serverless GPUs.

1. [Create an account on Beam](https://beam.cloud). It's free and you don't need a credit card.

2. Install the Beam CLI:

```bash
curl https://raw.githubusercontent.com/slai-labs/get-beam/main/get-beam.sh -sSfL | sh
```

3. Clone this example to your computer:

```python
beam create-app codeT5+
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
> "def print_hello_world():"

Example response: 
> "def print_hello_world():
        print("Hello World")
        print_hello_world()
    def print_hello_world_again(name):
        print("Hello World again")
        print("My name is " + name)
    print_hello_world_again("Juan")
    def print_hello_world_again_again(name):
        print("Hello World again again")
        print("My name is " + name)
    print_hello_world_again_again("Juan")
    def print_hello_world_again_again_again(name):
        print("Hello World again again again")
        print("My name is " + name)
    print_hello_world_again_again_again("Juan")
    def print_hello_world_again_again_again_again(name):
        print("Hello World again again again again")
        print("My name is " + name)
    print_hello_world_again_again_again_again("Juan")
    def print_hello_world_again_again_again_again_
