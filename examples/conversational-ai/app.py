"""
This example is a conversational AI app, using LangChain.

It's inspired by this post:
https://langchain.readthedocs.io/en/latest/modules/chains/combine_docs_examples/question_answering.html

You can run this example with this command:

beam run app.py:start_conversation -d '{"urls": ["https://www.nutribullet.com"], "query": "What are some use cases I can use this product for?"}'
"""

from beam import App, Runtime, Image


from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import UnstructuredURLLoader
from langchain.llms import OpenAI

# Add your OpenAI API Key to the Beam Secrets Manager:
# beam.cloud/dashboard/settings/secrets so that it is accessible through:
# os.environ["OPENAI_API_KEY"]


app = App(
    name="conversational-ai",
    runtime=Runtime(
        cpu=2,
        gpu="T4",
        memory="8Gi",
        image=Image(
            python_packages=[
                "langchain",
                "unstructured[csv]",
                "openai",
                "unstructured",
                "pdf2image",
                "tabulate",
            ],
        ),
    ),
)


# Deploys the function as a REST API
@app.rest_api()
def start_conversation(**inputs):
    # Grab inputs passed to the API
    urls = inputs["urls"]
    query = inputs["query"]

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
