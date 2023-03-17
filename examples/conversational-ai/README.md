## LangChain - Beam Template

This template demonstrates how to deploy LangChain apps using [Beam](https://beam.cloud).

This example implements a basic version of the [Question Answering guide](https://langchain.readthedocs.io/en/latest/modules/chains/combine_docs_examples/question_answering.html). [A more complete code walk-through can be found here](https://docs.beam.cloud/getting-started/langchain).

Note: You'll need accounts on [OpenAI](https://openai.com) and [PromptLayer](https://promptlayer.com) to run this example.

## Deploying on Beam

You can easily deploy your LangChain apps as web endpoints:

1. Create an account on [Beam](https://beam.cloud)
2. Download the CLI and Python-SDK. [Instructions here](https://docs.beam.cloud/getting-started/quickstart).
3. Add your `OPENAI_API_KEY` to the [Beam Secrets Manager](https://www.beam.cloud/dashboard/settings/secrets)
4. Add your `PROMPTLAYER_API_KEY` to the [Beam Secrets Manager](https://www.beam.cloud/dashboard/settings/secrets)
5. Download this template, and run `beam deploy app.py` from the working directory.

## Example Request

```cURL
 curl -X POST --compressed "https://beam.slai.io/cjm9u" \
   -H 'Authorization: Basic [ADD_YOUR_AUTH_TOKEN]' \
   -H 'Content-Type: application/json' \
   -d '{"query": "Give me a summary of the days news"}'
```

## Example Response

```cURL
{"pred":{"output_text":" Today's news includes stories about the Federal Reserve slowing interest rate increases, the funeral of a victim of police violence, the College Board stripping down its A.P. curriculum for African American studies, Biden and McCarthy discussing the debt limit, women being misled about menopause, an FBI search of Biden's vacation home, Tom Brady announcing his retirement, A.I. bots writing a column, Russia's involvement in the Ukraine war, and more."}}
```
