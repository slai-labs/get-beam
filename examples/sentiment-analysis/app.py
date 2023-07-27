from beam import App, Runtime, Image
from transformers import pipeline

app = App(
    name="sentiment-analysis",
    runtime=Runtime(
        cpu=4,
        memory="32Gi",
        image=Image(
            python_version="python3.9",
            python_packages=["transformers", "torch"],
        ),
    ),
)


@app.rest_api()
def predict_sentiment(**inputs):
    model = pipeline(
        "sentiment-analysis", model="siebert/sentiment-roberta-large-english"
    )

    result = model(inputs["text"], truncation=True, top_k=1)
    prediction = {i["label"]: i["score"] for i in result}

    print(prediction)

    return {"prediction": prediction}
