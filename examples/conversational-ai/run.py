import os
import requests
from pathlib import Path
from bs4 import BeautifulSoup

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import PromptLayerOpenAIChat


# Add your OpenAI API Key to the Beam Secrets Manager:
# beam.cloud/dashboard/settings/secrets
openai_api_key = os.environ["OPENAI_API_KEY"]

# We'll save our headlines to this path
file_path = Path("/workspace/transcript.txt")

# Download headlines from NYT
def download_headlines():
    res = requests.get("https://www.nytimes.com")
    soup = BeautifulSoup(res.content, "html.parser")
    # Grab all headlines
    headlines = soup.find_all("h3", class_="indicate-hover", string=True)
    parsed_headlines = []
    for h in headlines:
        parsed_headlines.append(h.get_text())

    # Write headlines to a text file
    with open(file_path, "w") as f:
        f.write(str(parsed_headlines))
        f.close()


# Answer questions about the headlines
def start_conversation(**inputs):
    # Grab the input from the API
    query = inputs["query"]

    if not file_path.exists():
        # Download headlines from nytimes.com and save to the file path above
        download_headlines()

    with open(file_path) as f:
        saved_file = f.read()
        # Split the text to conform to maximum number of tokens
        text_splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

        texts = text_splitter.split_text(saved_file)
        embeddings = OpenAIEmbeddings()
        docsearch = FAISS.from_texts(texts, embeddings)
        docs = docsearch.similarity_search(query)

        chain = load_qa_chain(
            PromptLayerOpenAIChat(openai_api_key=openai_api_key, pl_tags=["langchain"]),
            chain_type="stuff",
        )
        res = chain(
            {"input_documents": docs, "question": query}, return_only_outputs=True
        )
        print(res)
        return {"pred": res}


if __name__ == "__main__":
    # You can customize this query however you want:
    query = "What happened in sports?"
    start_conversation(query=query)
