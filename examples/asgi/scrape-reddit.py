from beam import App, Runtime, Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import requests

beam_app = App(
    "scrape-reddit",
    runtime=Runtime(
        cpu=1,
        memory="1Gi",
        image=Image(python_packages=["fastapi", "httpx"]),
    ),
)


# Define a FastAPI app
app = FastAPI()


# Pydantic model for a Reddit post
class RedditPost(BaseModel):
    title: str
    url: str
    upvotes: int


# Function to get top posts from Reddit
def get_top_posts(subreddit: str, limit: int = 5):  # -> List[RedditPost]:
    headers = {"User-Agent": "FastAPI Reddit Bot/0.1"}
    params = {"limit": limit}
    response = requests.get(
        f"https://www.reddit.com/r/{subreddit}/top/.json",
        headers=headers,
        params=params,
    )

    if response.status_code != 200:
        raise HTTPException(status_code=404, detail="Subreddit not found")

    top_posts = []

    for post in response.json()["data"]["children"]:
        data = post["data"]
        top_posts.append(
            RedditPost(title=data["title"], url=data["url"], upvotes=data["ups"])
        )

    return top_posts


# Endpoint to get top posts from a specific subreddit
@app.get("/r/{subreddit}/top", response_model=List[RedditPost])
def read_top_posts(subreddit: str, limit: Optional[int] = 1):
    return get_top_posts(subreddit, limit)


# Entrypoint to the app. When deployed, this HTTP endpoint will be publicly exposed to the internet.
@beam_app.asgi(authorized=False)
def handler():
    return app
