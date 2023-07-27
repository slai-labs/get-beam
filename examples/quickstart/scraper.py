"""
These packages don't necessarily need to be installed locally.
They will be added in the container image defined below.
"""
from beam import App, Runtime, Image, Output

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

app = App(
    name="web_scraper",
    runtime=Runtime(
        image=Image(
            python_packages=["requests", "beautifulsoup4"],
        ),
    ),
)


@app.run(outputs=[Output(path="results.txt")])
def scrape_wikipedia():
    url = "https://en.wikipedia.org/wiki/Main_Page"

    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    for link in soup.find_all("a", href=True):
        absolute_link = urljoin(url, link["href"])
        with open("results.txt", "a") as file:
            print(f"Found link: {absolute_link}")
            file.write(absolute_link + "\n")
