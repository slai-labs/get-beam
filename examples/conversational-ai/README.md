## LangChain - Beam Template

This template demonstrates how to build and deploy LangChain apps using [Beam](https://beam.cloud).

This example implements a basic version of the [Question Answering guide](https://langchain.readthedocs.io/en/latest/modules/chains/combine_docs_examples/question_answering.html). [A more complete code walk-through can be found here](https://docs.beam.cloud/examples/langchain).

Note: You'll need an API key from [OpenAI](https://openai.com) to run this example.

## Deploying on Beam

You can easily deploy your LangChain apps as web endpoints:

1. Create an account on [Beam](https://beam.cloud)
2. Download the CLI and Python-SDK. [Instructions here](https://docs.beam.cloud/getting-started/quickstart).
3. Add your `OPENAI_API_KEY` to the [Beam Secrets Manager](https://www.beam.cloud/dashboard/settings/secrets)
4. Download this template, and run `beam deploy app.py` from the working directory.

## Example Request

```cURL
 curl -X POST --compressed "https://beam.slai.io/cjm9u" \
   -H 'Authorization: Basic [ADD_YOUR_AUTH_TOKEN]' \
   -H 'Content-Type: application/json' \
   -d '{"urls": "[\"https://apple.com\"]", "query": "What kind of products does this company sell?"}'
```

## Example Response

```cURL
{"pred":{"output_text":" This company sells iPhones, Apple Watches, iPads, MacBook Pros, Apple Trade In, Apple Card, and Apple TV+."}}
```
