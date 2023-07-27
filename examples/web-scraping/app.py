"""
Run this example: beam run app.py:scrape_nyt
"""
from beam import App, Runtime, Image

import requests
from bs4 import BeautifulSoup
from transformers import pipeline

app = App(
    name="web-scraper",
    runtime=Runtime(
        cpu=1,
        memory="8Gi",
        image=Image(
            python_version="python3.8",
            python_packages=["bs4", "transformers", "torch"],
        ),
    ),
)


@app.run()
def scrape_nyt():
    res = requests.get("https://www.nytimes.com")
    soup = BeautifulSoup(res.content, "html.parser")
    # Grab all headlines
    headlines = soup.find_all("h3", class_="indicate-hover", text=True)

    total_headlines = len(headlines)
    negative_headlines = 0

    # Iterate through each headline
    for h in headlines:
        title = h.get_text()
        print(title)
        sentiment = predict_sentiment(title)

        print(sentiment)

        if sentiment.get("NEGATIVE") > sentiment.get("POSITIVE"):
            negative_headlines += 1

    print(f"{negative_headlines} negative headlines / {total_headlines} total")


def predict_sentiment(title):
    model = pipeline(
        "sentiment-analysis", model="siebert/sentiment-roberta-large-english"
    )
    result = model(title, truncation=True, top_k=2)
    prediction = {i["label"]: i["score"] for i in result}

    return prediction


if __name__ == "__main__":
    scrape_nyt()
