from beam import App, Runtime, Image

app = App(
    name="preload-models",
    runtime=Runtime(),
)


def load_models():
    # This runs exactly once when the container first starts
    model = {"your_model": ""}
    print("Loader running!")
    return model


@app.rest_api(loader=load_models)
def predict(**inputs):
    # The loaded model is passed in using the `context` argument
    loaded_model = inputs["context"]
    print(f"Loaded Model: {loaded_model}")
