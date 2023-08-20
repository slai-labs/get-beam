import base64
import requests
from requests.auth import HTTPBasicAuth

# Beam Credentials / App ID
client_id = "YOUR_CLIENT_ID"
client_secret = "YOUR_CLIENT_SECRET"
beam_app_id = "YOUR_APP_ID"

# Example audio file, for testing purposes
mp3_filepath = "./example.wav"

# Send the audio file as base64
encode_audio = base64.b64encode(open(mp3_filepath, "rb").read()).decode("UTF-8")

data = {
    "audio_file": encode_audio,
    "model": "small",
}

headers = {
    "Accept": "*/*",
    "Connection": "keep-alive",
    "Content-Type": "application/json",
}

requests.post(
    f"https://apps.beam.cloud/{beam_app_id}",
    auth=HTTPBasicAuth(client_id, client_secret),
    headers=headers,
    json=data,
)
