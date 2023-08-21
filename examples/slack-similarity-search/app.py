import os
import json

from beam import App, Runtime, Image
from metal_sdk.metal import Metal

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError


# Add your own Slack token and Channel ID to the secrets manager in Beam:
# https://www.beam.cloud/dashboard/settings/secrets
slack_token = os.environ["SLACK_TOKEN"]
channel_id = os.environ["SLACK_CHANNEL_ID"]

app = App(
    name="slack-similarity-search",
    runtime=Runtime(
        cpu=1,
        memory="8Gi",
        image=Image(
            python_packages=[
                "metal_sdk",
                "slack-sdk",
            ],
        ),
    ),
)

metal = Metal(
    os.environ["METAL_API_KEY"],
    os.environ["METAL_CLIENT_ID"],
    os.environ["METAL_INDEX_ID"],
)


# Deploys the function as a REST API
@app.rest_api()
def search_conversations(**inputs):
    query = inputs["query"]
    metal_response = metal.search(
        {
            "text": query,
        },
        index_id=os.environ["METAL_INDEX_ID"],
        limit=5,
    )

    response = json.loads(metal_response.content)

    # Get permalink to the relevant Slack conversation
    client = WebClient(slack_token)
    permalink = client.chat_getPermalink(
        channel=channel_id, message_ts=response["data"][0]["metadata"]["ts"]
    )

    payload = {
        "permalink": permalink["permalink"],
        "your_query": query,
        "original_message": response["data"][0]["text"],
    }

    print(payload)

    return {"results": payload}


# Retrieve messages from Slack, using the Channel ID set above
def scrape_slack():
    client = WebClient(slack_token)
    try:
        all_messages = []
        cursor = None

        while True:
            response = client.conversations_history(channel=channel_id, cursor=cursor)
            all_messages.extend(response["messages"])
            cursor = response.get("response_metadata", {}).get("next_cursor")

            if not cursor:
                break

        return all_messages

    except SlackApiError as e:
        print(f"Error: {e.response['error']}")
        return None


# Run this manually to scrape slack and index the messages in Metal
def populate_index():
    conversation_messages = scrape_slack()
    messages = []
    # Loop through all messages, Metal has a limit of 100 records
    for message in conversation_messages[:100]:
        if len(message["text"]) > 3:
            payload = {
                "text": message["text"],
                "index": os.environ["METAL_INDEX_ID"],
                "metadata": {
                    "user": message["user"],
                    "type": message["type"],
                    "ts": message["ts"],
                },
            }
            messages.append(payload)

    # Save messages to Metal, in bulk
    metal.index_many(messages)


if __name__ == "__main__":
    # populate_index()
    search_conversations(query="how does billing work?")
