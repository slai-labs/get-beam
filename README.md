<p align="center">
<img alt="Logo" src="https://slai-demo-datasets.s3.amazonaws.com/git-header.png"/ width="1000">
</p>

<h3 align="center">
Beam is a cloud platform where you can provision infrastructure, develop on remote runtimes from your local machine, and deploy apps as serverless functions — without leaving your IDE
</h3>

<p align="center">
<a href="https://join.slack.com/share/enQtNDMwOTExNDI3NTE0MS1hZTlhNWJlMmJjZmExY2MzZGZhMTg4MWJhNzEwZTc5YTQwMjM1MDY5NDY1NThlYjA0NzM1NTQzYzI3MjgzZjQx"><img src="https://img.shields.io/badge/join-Slack-yellow"/></a>
<a href="https://docs.slai.io/beam"><img src="https://img.shields.io/badge/docs-quickstart-blue"/></a>


# Features 

## 📦 Setup remote development environments in code

Configure your runtime in Python - tell us how many GPUs you need and which libraries you want installed, and Beam will spawn a remote environment for you.

<video controls muted autoplay loop>
  <source src="https://slai-demo-datasets.s3.amazonaws.com/define-runtime.mp4" type="video/mp4"></source>
</video>

## 🛰 A local development experience on remote hardware

You can write and run your code locally - except when you enter your shell, your code will run on Beam instead of your local machine.

<video controls muted autoplay loop>
  <source src="https://slai-demo-datasets.s3.amazonaws.com/develop-locally.mp4" type="video/mp4"></source>
</video>

## 🚀 Deploy apps as serverless functions

Deploy your apps as serverless REST APIs, scheduled cron jobs, or webhooks - all in just four lines of Python.

<video controls muted autoplay loop>
  <source src="https://slai-demo-datasets.s3.amazonaws.com/deploy.mp4" type="video/mp4"></source>
</video>


# Installation

```bash
curl https://raw.githubusercontent.com/slai-labs/get-beam/main/get-beam.sh -sSfL | sh
```

# Getting Started

1. [Create a Slai account](https://slai.io) and grab your API keys from the [settings page](https://www.slai.io/beam/apps/settings/api-keys)
2. Configure your credentials (you'll be prompted to enter your API keys)

```bash
beam configure
```

3. Create a virtual env in the directory you want to work in

```bash
python3 -m virtualenv .venv
```

4. Source the virtual env

```bash
source .venv/bin/activate
```

5. Install Beam SDK

```bash
pip install beam-sdk
```

# Workflow

**Develop**

```bash
beam start <myapp.py>
```

**Deploy**

```bash
beam deploy <myapp.py>
```

# Documentation

[Check out our complete documentation here »](https://docs.slai.io/beam)

