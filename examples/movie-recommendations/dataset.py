"""
Dataset loader for MovieLens
"""

import math
import random
from torch.utils.data import Dataset
import pandas as pd


class MovieLensDataset(Dataset):
    """
    This class implements the PyTorch dataset interface
    """

    def __init__(
        self,
        path: str,
        train: bool = True,
        train_size: float = 0.8,
        negatives: int = 100,
    ):
        df = pd.read_csv(path)

        self.users = []
        self.movies = []
        self.labels = []
        self.user_movies = {}

        users = df["userId"].unique()
        movies = df["movieId"].unique()

        for u in users:
            # Split train/test temporally for each user
            user_movies = df[df["userId"] == u]
            user_movies = user_movies.sort_values(by="timestamp")
            self.user_movies[u] = set(user_movies.movieId)

            train_count = int(math.floor(train_size * len(user_movies)))
            if train:
                user_movies = user_movies.head(train_count)
            else:
                user_movies = user_movies.tail(len(user_movies) - train_count)

            used = set()

            # Positive samples
            for idx, row in user_movies.iterrows():
                self.users.append(u)
                m = int(row.movieId)
                self.movies.append(m)
                self.labels.append(1)
                used.add(m)

            # Negative samples
            # Assume that any rating or interaction between a user and a movie is a binary “1”.
            neg = 0
            while neg < negatives:
                m = random.choice(movies)
                if m in used:
                    continue
                self.users.append(u)
                self.movies.append(m)
                self.labels.append(0)
                used.add(m)
                neg += 1

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.labels[idx]
