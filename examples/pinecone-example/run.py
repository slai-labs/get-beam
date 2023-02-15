"""
When building these generative QA apps, you have two data sources used to initalize the model:
One data source is the prompts, the other is the source data. 

1. Initialize the index: Pick a datasource and initialize it in Pinecone, which stores the vectors 
2. Setup a retriever: this generates embeddings for all the vectors and the questions, so that the questions and vectors are close together in vector space 
3. Generate the embeddings and 'upsert' (upload) them to Pinecone
"""

import os
from tqdm.auto import tqdm
from datasets import load_dataset
import pinecone

from sentence_transformers import SentenceTransformer
from transformers import pipeline


def get_or_create_index():
    # Add your Pinecone credentials to the Beam Secrets Manager
    # If you don't have a Pinecone account, you can create one here: app.pinecone.io
    pinecone.init(
        api_key=os.environ["PINECONE_API_KEY"],
        environment=os.environ["PINECONE_ENVIRONMENT"],
    )

    index_name = "extractive-question-answering"
    # Check if the index already exists
    if index_name not in pinecone.list_indexes():
        # Create the index if it does not exist
        pinecone.create_index(index_name, dimension=384, metric="cosine")

    # Connect to index
    index = pinecone.Index(index_name)
    return index


def initialize_retriever():
    # Load the squad dataset into a pandas dataframe
    df = load_dataset("squad", split="train").to_pandas()
    # Select only title and context column
    df = df[["title", "context"]]
    # Drop rows containing duplicate context passages
    df = df.drop_duplicates(subset="context")
    # Load the retriever model from huggingface model hub
    retriever = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

    # We will use batches of 64
    batch_size = 64

    for i in tqdm(range(0, len(df), batch_size)):
        # Find end of batch
        i_end = min(i + batch_size, len(df))
        # Extract batch
        batch = df.iloc[i:i_end]
        # Generate embeddings for batch
        emb = retriever.encode(batch["context"].tolist()).tolist()
        # Get metadata
        meta = batch.to_dict(orient="records")
        # Create unique IDs
        ids = [f"{idx}" for idx in range(i, i_end)]
        # Add all to upsert list
        to_upsert = list(zip(ids, emb, meta))
        # Upsert/insert these records to pinecone
        index = get_or_create_index()
        _ = index.upsert(vectors=to_upsert)

    # Check that we have all vectors in index
    index.describe_index_stats()


def get_reader():
    model_name = "deepset/electra-base-squad2"
    # Load the reader model into a question-answering pipeline
    reader = pipeline(
        tokenizer=model_name,
        model=model_name,
        task="question-answering",
        cache_dir="./cached_models",  # Cache model in the Persistent Volume mounted in app.py
    )
    return reader


def get_context(**inputs):
    # Load the question as a kwarg, passed in from the API
    question = inputs["question"]

    index = get_or_create_index()
    retriever = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
    # Generate embeddings for the question
    question_embeddings = retriever.encode([question]).tolist()
    # Search Pinecone index for context passage with the answer
    pinecone_response = index.query(question_embeddings, top_k=1, include_metadata=True)
    # Extract the context passage from pinecone search result
    context = [i["metadata"]["context"] for i in pinecone_response["matches"]]
    return context


# Extracts answer from the context passage
def extract_answer(**inputs):
    # Load the question as a kwarg, passed in from the API
    question = inputs["question"]
    context = inputs["context"]

    results = []

    reader = get_reader()
    for c in context:
        # Feed the reader the question and contexts to extract answers
        answer = reader(question=question, context=c)
        # Add the context to answer dict for printing both together
        answer["context"] = c
        results.append(answer)
    # Sort the result based on the score from reader model
    sorted_result = sorted(results, key=lambda x: x["score"], reverse=True)
    return sorted_result


# Inference
def answer_question(**inputs):
    question = inputs["question"]

    context = get_context(question=question)
    res = extract_answer(question=question, context=context)

    for r in res:
        answer = r["answer"]
        context = r["context"]

    return {"answer": answer, "context": context}


if __name__ == "__main__":
    # Initialize Pinecone
    # initialize_retriever()
    question = "What are the first names of the men that invented youtube?"
    answer_question(question=question)