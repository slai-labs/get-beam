import requests
from bs4 import BeautifulSoup

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

# We'll save our headlines to this path
file_path = "/workspace/transcript.txt"


# Download headlines from NYT
def download_headlines():
    res = requests.get("https://www.nytimes.com")
    soup = BeautifulSoup(res.content, "html.parser")
    # Grab all headlines
    headlines = soup.find_all("h3", class_="indicate-hover", text=True)
    parsed_headlines = []
    for h in headlines:
        parsed_headlines.append(h.get_text())

    print(parsed_headlines)

    # Write headlines to a text file
    with open(file_path, "w") as f:
        f.write(str(parsed_headlines))
        f.close()


# Answer questions about the headlines
def start_conversation(**inputs):
    # Grab the input from the API
    query = inputs["query"]
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

        chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
        res = chain(
            {"input_documents": docs, "question": query}, return_only_outputs=True
        )
        print(res)
        return {"pred": res}


if __name__ == "__main__":
    # # You can customize this query however you want:
    # # For example: What happened in Washington today?
    query = "Give me a summary of today's news"
    start_conversation(query=query)
