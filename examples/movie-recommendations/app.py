"""
This example demonstrates a basic movie recommendation system. The following capabilities are demonstrated:

1. Training a model using the MovieLens dataset: https://grouplens.org/datasets/movielens/
2. Saving the trained model to a Persistent Volume
3. Retrieving the trained model from a Persistent Volume during inference
4. Deploying a REST API that accepts a user ID and returns customized recommendations for that user
"""

import beam

"""
This is the runtime our code will run in. You can add a GPU if you like, but it's not really necessary for this example.
"""
app = beam.App(
    name="movie-recommendation-example",
    cpu=4,
    memory="8Gi",
    python_version="python3.8",
    python_packages=["numpy", "torch", "pandas", "matplotlib"],
)


"""
Deploy an endpoint that takes (1) a user ID and (2) the number of recommendations to return per user.
"""
app.Trigger.RestAPI(
    inputs={
        "user_id": beam.Types.Float(),
        "number_of_recommendations": beam.Types.Float(),
    },
    outputs={
        "recommendations": beam.Types.Json(),
    },
    handler="run.py:run_inference",
)

"""
We're going to mount a Persistent Volume, which is a writable data store

- During training, we will save our models to this volume
- During inference, we will retrieve our trained model 

We can access the volume at this path: /volumes/trained_models
"""
app.Mount.PersistentVolume(name="trained_models", app_path="./trained_models")
