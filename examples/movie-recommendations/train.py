"""
Trains a neural collaborative filtering recommender model on the MovieLens dataset

Start a training job:

beam run train.py:run_training_pipeline
"""
from beam import App, Runtime, Image, Volume

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


from dataset import MovieLensDataset

device = "cpu"

training_app = App(
    name="movie-recommendation-training",
    runtime=Runtime(
        cpu=4,
        memory="16Gi",
        image=Image(
            python_version="python3.8",
            python_packages=["numpy", "torch", "pandas", "matplotlib"],
        ),
    ),
    volumes=[Volume(name="trained_models", path="./trained_models")],
)


class NCF(nn.Module):
    """
    Use an embedding layer for both the user and movie, to compress the
    respective one-hot encoded vectors into rich, compact representations
    that are easier to model.
    """

    def __init__(self, num_users, num_items):
        super().__init__()
        # User embedding layer
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=8)
        # Movie embedding layer
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=8)
        self.fc1 = nn.Linear(in_features=16, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.output = nn.Linear(in_features=32, out_features=1)

    def forward(self, user, item):
        # Embedding
        u = self.user_embedding(user)
        i = self.item_embedding(item)
        x = torch.cat([u, i], dim=-1)

        # Dense layers
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        # Output
        x = self.output(x)
        x = F.sigmoid(x)
        return x


def load_model():
    # Load MovieLens data
    dataset_train = MovieLensDataset(
        "./data/ratings.csv",
        train=True,
        train_size=0.8,
        negatives=128,
    )
    dataset_test = MovieLensDataset(
        "./data/ratings.csv",
        train=False,
        train_size=0.8,
        negatives=32,
    )
    dataset_test_positives = MovieLensDataset(
        "./data/ratings.csv",
        train=False,
        train_size=0.8,
        negatives=0,
    )
    print(
        "Loaded {} training samples and {} test samples".format(
            len(dataset_train), len(dataset_test)
        )
    )

    # Setup model
    num_users = max(dataset_train.users) + 1
    num_movies = max(dataset_train.movies) + 1

    model = NCF(num_users, num_movies).to(device)
    return model


def train():
    # load movielens data
    dataset_train = MovieLensDataset(
        "./data/ratings.csv",
        train=True,
        train_size=0.8,
        negatives=128,
    )
    dataset_test = MovieLensDataset(
        "./data/ratings.csv",
        train=False,
        train_size=0.8,
        negatives=32,
    )
    dataset_test_positives = MovieLensDataset(
        "./data/ratings.csv",
        train=False,
        train_size=0.8,
        negatives=0,
    )
    print(
        "Loaded {} training samples and {} test samples".format(
            len(dataset_train), len(dataset_test)
        )
    )

    loader_train = DataLoader(dataset_train, batch_size=1024, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=1024)
    loader_test_positives = DataLoader(dataset_test_positives, batch_size=1024)

    unique_movies = set(dataset_train.movies)
    unique_users = set(dataset_train.users)

    # Setup model
    model = load_model()
    optimizer = torch.optim.Adam(model.parameters())

    # *** Training ***
    for epoch in range(0, 20):
        # To begin the training process, the model weights are randomly initialized
        # We use an Adam optimizer with a binary cross entropy loss function to minimize error in predicting interactions between users and movies.
        model.train()
        train_loss = 0
        for batch_idx, (user, movie, label) in enumerate(loader_train):
            user, movie, label = user.to(device), movie.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(user, movie)
            loss = F.binary_cross_entropy(output, label.view(-1, 1).float())
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        train_loss /= len(loader_train)
        print("Train epoch: {}, avg loss: {:.6f}".format(epoch, train_loss))

        # Test
        model.eval()
        test_loss = 0
        hits = 0
        with torch.no_grad():
            # Loss
            for user, movie, label in loader_test:
                user, movie, label = user.to(device), movie.to(device), label.to(device)
                output = model(user, movie)
                test_loss += F.binary_cross_entropy(
                    output, label.view(-1, 1).float()
                ).item()

            # Calculates hit rate -- basically, given N total samples, including 1 positive sample, what is the probability
            # that the positive sample will appear in the top K results. We can refer to this as “hit rate @ K / N”.

            # Hit rate @ 10
            k = 10
            total = 1000
            hit_thresholds = {}
            for u in unique_users:
                negatives = random.sample(
                    [
                        m
                        for m in unique_movies
                        if m not in dataset_test_positives.user_movies[u]
                    ],
                    total,
                )
                negatives = torch.tensor(negatives).to(device)
                user = torch.tensor([u] * total).to(device)
                output = model(user, negatives)
                top_k = torch.topk(output.flatten(), k)
                hit_thresholds[u] = top_k.values[k - 1].item()

            for user, movie, label in loader_test_positives:
                user, movie, label = user.to(device), movie.to(device), label.to(device)
                output = model(user, movie)
                for u, o in zip(user, output):
                    if o.item() > hit_thresholds[u.item()]:
                        hits += 1

        test_loss /= len(loader_test)
        hit_rate = hits / len(dataset_test_positives)

        print(
            "Test set: avg loss: {:.4f}, hit rate: {}/{} ({:.2f}%)\n".format(
                test_loss,
                hits,
                len(dataset_test_positives),
                100.0 * hit_rate,
            )
        )

    return model


@training_app.run()
def run_training_pipeline():
    # Trains a model and saves the state_dict to the persistent volume
    trained_model = train()
    persistent_volume_path = "/volumes/trained_models/model.pt"
    torch.save(trained_model.state_dict(), persistent_volume_path)
