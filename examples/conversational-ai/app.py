import beam

app = beam.App(
    name="conversational-ai",
    cpu=8,
    gpu=1,
    memory="32Gi",
    python_packages=["bs4", "openai", "langchain", "faiss-cpu"],
)

# Triggers determine how your app is deployed
app.Trigger.RestAPI(
    inputs={"query": beam.Types.String()},
    outputs={"pred": beam.Types.Json()},
    handler="run.py:start_conversation",
)

app.Output.File(name="transcript", path="transcript.txt")
