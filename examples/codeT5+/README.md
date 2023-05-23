# CodeT5+ deployment with Beam

[DLite](https://github.com/salesforce/CodeT5/tree/main/CodeT5+) can be deployed on Beam for development
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

## Deploying codeT5+-6B

Create a local copy of the project by running:

```beam create-app codeT5+```

And deploy it using:

```beam deploy app.py```

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
