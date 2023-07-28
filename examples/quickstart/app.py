from beam import App, Runtime

app = App(name="beam-quickstart", runtime=Runtime())


@app.rest_api()
def run():
    print("🔮 This is running remotely on Beam!")
    return {"message": "Nice work! Check out our docs for more: https://docs.beam.cloud/getting-started/quickstart"}
