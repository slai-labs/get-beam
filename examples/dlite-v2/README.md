# DLite deployment with Beam

[DLite](https://huggingface.co/aisquared/dlite-v2-1_5b) can be run on the cloud with a single command, using Beam.

1. [Create an account on Beam](https://beam.cloud). It's free and you don't need a credit card.

2. Install the Beam CLI:

```bash
curl https://raw.githubusercontent.com/slai-labs/get-beam/main/get-beam.sh -sSfL | sh
```

3. Clone this example to your computer:

```python
beam create-app dlite-v2
```

4. Deploy and run inference:

```python
beam deploy app.py
```

This example is called through a task queue. Task queues are used for deploying
functions that run asynchronously on Beam. Here, the task queue takes a prompt
as one of its input fields. An example prompt and the real response from the
model are given below.

Example prompt:
> Once upon a time

Example response: 
> Once upon a time in a kingdom not far distant, there was a king, and amid the festivities of his court a fair was going on. As the time for the Fair drew near, the king selected five youths to be his Twelfthlings. These young men were then marshaled into a big boat and taken on a journey. They were told that they would find fame and fortune if they could sail into the fog and then swim back to land. They started their journey and soon reached a spot where the water was relatively clear. They thanked the lord of the land for the excellent weather, but were then told that it would be much better to remain in the thick of the fog until they reached land. They could then swim back to land.
