"""
"""
import beam

# App configuration -- this is the compute the app will run on
app = beam.App(
    name="pinecone-example",
    cpu=8,
    gpu="A10G",
    memory="32Gi",
    python_packages=[
        "pinecone-client",
        "sentence-transformers",
        "torch",
        "datasets",
    ],
)

# Add a REST API Trigger to deploy this app as a web endpoint
app.Trigger.RestAPI(
    inputs={
        # Takes a question as input -- this is passed as a keyword argument to the handler function
        "question": beam.Types.String()
    },
    outputs={
        "answer": beam.Types.String(),
        "context": beam.Types.String(),
    },  # Returns an answer
    handler="run.py:answer_question",  # This is the function that will be run when the endpoint is invoked
)

# This volume is used to cache the Huggingface model
app.Mount.PersistentVolume(app_path="./cached_models", name="cached-models")
