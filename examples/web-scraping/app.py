import beam

app = beam.App(
    name="web-scraper",
    cpu=4,
    memory="4Gi",
    gpu=0,
    python_version="python3.8",
    python_packages=["bs4", "transformers", "torch"],
)