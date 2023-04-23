import os

from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import UnstructuredURLLoader
from langchain.llms import OpenAI


# Add your OpenAI API Key to the Beam Secrets Manager:
# beam.cloud/dashboard/settings/secrets
openai_api_key = os.environ["OPENAI_API_KEY"]


# Answer questions about the URLs provided
def start_conversation(**inputs):
    # Grab the input from the API
    query = inputs["query"]
    urls = inputs["urls"]

    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    doc_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    docs = doc_splitter.split_documents(data)

    chain = load_qa_chain(
        OpenAI(temperature=0),
        chain_type="stuff",
    )
    res = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
    print(res)
    return {"pred": res}


if __name__ == "__main__":
    # You can customize this query however you want:
    urls = ["https://www.nutribullet.com"]
    query = "What are some use cases I can use this product for?"
    start_conversation(urls=urls, query=query)
