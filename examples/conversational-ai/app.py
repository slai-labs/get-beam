"""
This example is a conversational AI app, using LangChain.

Users can ask queries about the day's New York Times headlines, and the conversational AI will answer questions about them.

It's inspired by this post:
https://langchain.readthedocs.io/en/latest/modules/chains/combine_docs_examples/question_answering.html
"""
import beam

app = beam.App(
    name="conversational-ai",
    cpu=8,
    gpu="A10G",
    memory="32Gi",
    python_packages=[
        "langchain",
        "openai",
        "unstructured",
    ],
)

# The REST API trigger exposes the app as a REST endpoint when deployed
app.Trigger.RestAPI(
    inputs={"query": beam.Types.String(), "urls": beam.Types.Json()},
    outputs={"pred": beam.Types.Json()},
    handler="run.py:start_conversation",
)
