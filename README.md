# 📡 Beam
**Beam is a rapid prototyping framework for data projects.** _Releases of beam binaries will be listed here._

# Installation

```bash
curl https://raw.githubusercontent.com/slai-labs/get-beam/main/get-beam.sh -sSfL | sh
```

# Getting Started

1. [Create a Slai account](https://slai.io) and grab your API keys from the [settings page](https://slai.io/settings/api-keys)
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

# Further Reading

[Check out our complete documentation here.](https://docs.slai.io)
