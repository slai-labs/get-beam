from transformers import pipeline


def predict_sentiment(**inputs):
    model = pipeline(
        "sentiment-analysis", model="siebert/sentiment-roberta-large-english"
    )
    result = model(inputs["text"], truncation=True, top_k=1)
    prediction = {i["label"]: i["score"] for i in result}

    print(prediction)

    return prediction