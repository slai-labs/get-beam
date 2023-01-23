"""
This function returns the top 'N' unseen movie recommendations for a specific user. 

1. Loads the user viewing history
2. Filters out any previously viewed movies
3. Scores all unseen movie candidates
4. Returns the top N results.
"""

import torch
import pandas as pd

from dataset import MovieLensDataset
from train import load_model

# The path where we can retrieve the trained model
persistent_volume_path = "/volumes/trained_models/model.pt"


def run_inference(**inputs):
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

    # Returns top N unseen recommendations
    return {"recommendations": recs}


if __name__ == "__main__":
    prediction = run_inference(user_id=42, number_of_recommendations=10)
    print(prediction)
