import beam

app = beam.App(
    name="sentiment-analysis",
    cpu=4,
    memory="4Gi",
    gpu=0,
    apt_install=[],
    python_version="python3.9",
    python_packages=["transformers", "torch"],
)

app.Trigger.RestAPI(
    inputs={"text": beam.Types.String()},
    outputs={"prediction": beam.Types.String()},
    handler="inference.py:predict_sentiment",
)