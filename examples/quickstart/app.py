import beam

app = beam.App(
    name="hello-world",
    cpu=1,
    memory="4Gi",
    python_version="python3.8",
    python_packages=[],
)

app.Trigger.RestAPI(
    inputs={"text": beam.Types.StringType()},
    outputs={
        "response": beam.Types.StringType(),
    },
    handler="run.py:hello_world",
)
