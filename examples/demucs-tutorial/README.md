# Demucs Example

This example takes an audio file as input and returns a .zip file with separated audio samples.

# Pre-requisites 

1. Make sure you have [Beam](https://beam.cloud) installed: `curl https://raw.githubusercontent.com/slai-labs/get-beam/main/get-beam.sh -sSfL | sh`
2. Clone this repo and `cd` into the directory

# Quickstart

1. Test the API: `beam serve app.py`. You can make any desired changes to the code, and Beam will automatically reload the remote server each time you update your application code. 
> Note: Any updates to compute requirements, python packages, or shell commands will require you to manually restart the dev session)
2. Deploy the API: `beam deploy app.py`

# Calling the API

## Example Request

```sh
curl -X POST \
-H 'Authorization: Basic [YOUR AUTH TOKEN]' \
-H 'Content-Type: application/json' \
-d '{"audio":"https://www2.cs.uic.edu/~i101/SoundFiles/CantinaBand60.wav"}'
```
## Example Response

This run async, so a task ID is returned: 

```sh
{"task_id":"10c21544-5c1f-4168-a1ac-e692b4b410dd"}
```

# Retrieve Output Files

You can monitor the status of the request through the web dashboard or `/task` API.

```sh
curl -X GET \
  --header "Content-Type: application/json" \
  --user "{CLIENT_ID}:{CLIENT_SECRET}" \
  "https://api.beam.cloud/v1/task/{TASK_ID}/status/"
```

Once the task is finished, this call will return a pre-signed to download audio files that were generated.

```json
{
  "task_id": "10c21544-5c1f-4168-a1ac-e692b4b410dd",
  "started_at": "2023-09-30T19:43:25.668303Z",
  "ended_at": "2023-09-30T19:44:02.017401Z",
  "outputs": {
    "processed/htdemucs/audio": {
      "path": "processed/audio",
      "url": "http://data.beam.cloud/outputs/hw6hx-0001/10c21544-5c1f-4168-a1ac-e692b4b410dd/output.zip"
    }
  }
}
```

# Further Reading

* [Add Autoscaling to your API](https://docs.beam.cloud/deployment/autoscaling)
* [Setup Webhook Callbacks](https://docs.beam.cloud/deployment/callbacks)
* [How Pricing Works](https://docs.beam.cloud/account/pricing-and-billing)