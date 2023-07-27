"""
This example demonstrates a basic movie recommendation system. The following capabilities are demonstrated:

1. Training a model using the MovieLens dataset: https://grouplens.org/datasets/movielens/
2. Saving the trained model to a Persistent Volume
3. Retrieving the trained model from a Persistent Volume during inference
4. Deploying a REST API that accepts a user ID and returns customized recommendations for that user

**Run inference**

beam run app.py:run_inference -d '{"user_id": 42, "number_of_recommendations": 10}'

**Deploy this as a REST API**

beam deploy app.py:run_inference 
"""

from beam import App, Runtime, Image, Volume

import torch
import pandas as pd

from dataset import MovieLensDataset
from train import load_model

# The path where we can retrieve the trained model
persistent_volume_path = "/volumes/trained_models/model.pt"


"""
This is the runtime our code will run in. You can add a GPU if you like, but it's not really necessary for this example.
"""
inference_app = App(
    name="movie-recommendation-example",
    runtime=Runtime(
        cpu=4,
        memory="8Gi",
        image=Image(
            python_version="python3.8",
            python_packages=["numpy", "torch", "pandas", "matplotlib"],
        ),
    ),
    volumes=[Volume(name="trained_models", path="./trained_models")],
)


# Deploy an endpoint that takes (1) a user ID and (2) the number of recommendations to return per user.
@inference_app.rest_api()
def run_inference(**inputs):
    """
    This function returns the top 'N' unseen movie recommendations for a specific user.

    1. Loads the user viewing history
    2. Filters out any previously viewed movies
    3. Scores all unseen movie candidates
    4. Returns the top N results.
    """
    user_id = int(inputs["user_id"])
    number_of_recommendations = int(inputs["number_of_recommendations"])

    dataset = MovieLensDataset("./data/ratings.csv", train_size=0, negatives=0)
    movies = pd.read_csv("./data/movies.csv", index_col="movieId")

    # Gather all items that user has not interacted with
    unseen = torch.tensor(
        [m for m in movies.index if m not in dataset.user_movies[user_id]]
    )

    # Load trained model
    model = load_model()
    model.load_state_dict(torch.load(str(persistent_volume_path)))
    model.eval()

    # Predict recommendation scores
    pred = model(torch.tensor([user_id] * len(unseen)), unseen)

    top_k = torch.topk(pred.flatten(), number_of_recommendations)
    # Format scores usable results
    recs = []
    for i, score in zip(top_k.indices, top_k.values):
        m = unseen[i].item()
        recs.append(
            {
                "title": movies.loc[m].title,
                "genres": movies.loc[m].genres,
                "movie_id": m,
                "score": score.item(),
            }
        )
    
    print(f'recommendations: {recs}')

    # Returns top N unseen recommendations
    return {"recommendations": recs}


if __name__ == "__main__":
    prediction = run_inference(user_id=42, number_of_recommendations=10)
    print(prediction)
