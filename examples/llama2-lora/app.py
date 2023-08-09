from beam import App, Runtime, Image, Volume
from finetune import train

app = App(
    "llama-lora",
    runtime=Runtime(
        cpu=4,
        memory="32Gi",
        gpu="A10G",
        image=Image(
            python_version="python3.10",
            python_packages="requirements.txt",
        ),
    ),
    volumes=[Volume(name="yahma", path="./yahma")],
)


@app.schedule(when="every 24h")
def train_model():
    train(base_model="decapoda-research/llama-7b-hf")


if __name__ == "__main__":
    train_model()
