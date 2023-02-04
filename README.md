<p align="center">
<img alt="Logo" src="https://slai-demo-datasets.s3.amazonaws.com/git-header.png"/ width="1000">
</p>

<h4 align="center">
Beam is a cloud platform where you can provision infrastructure, develop on remote runtimes from your local machine, and deploy apps as serverless functions — without leaving your IDE
</h4>

<p align="center">
<a href="https://beam-89x5025.slack.com/join/shared_invite/zt-1jf3z8c01-ZXF6pPSCYdosaJ74__jaGw#/shared-invite/email"><img src="https://img.shields.io/badge/join-Slack-yellow"/></a>
<a href="https://docs.beam.cloud"><img src="https://img.shields.io/badge/docs-quickstart-blue"/></a>


# Features

### 📦 Setup remote development environments in code

Configure your runtime in Python - tell us how many GPUs you need and which libraries you want installed, and Beam will spawn a remote environment for you.

https://user-images.githubusercontent.com/10925686/199524970-ecd3d1a6-df4f-4dc1-ad21-1cc412a15673.mp4

### 🛰 Develop locally on remote hardware

You can write and run your code locally - except when you enter your shell, your code will run on Beam instead of your local machine.

https://user-images.githubusercontent.com/10925686/199525014-65bface8-589c-46b3-b742-8df41d10e981.mp4

### 🚀 Deploy apps as serverless functions

Deploy your apps as serverless REST APIs, scheduled cron jobs, or webhooks - all in just four lines of Python.

https://user-images.githubusercontent.com/10925686/199525037-1c246d7b-05af-41f1-8027-89e1ffbab0ed.mp4

# Installation

```bash
curl https://raw.githubusercontent.com/slai-labs/get-beam/main/get-beam.sh -sSfL | sh
```

# Getting Started

1. [Create an account on Beam](https://beam.cloud) and grab your API keys from the [settings page](https://www.beam.cloud/dashboard/settings/api-keys)
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

[Check out our complete documentation here »](https://docs.beam.cloud)
