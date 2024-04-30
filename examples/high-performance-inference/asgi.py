import asyncio

from beam import App, Image, Runtime
from fastapi import FastAPI, Response, WebSocket

app = App(
    "ws-server",
    runtime=Runtime(
        cpu=1,
        memory="1Gi",
        image=Image(python_packages=["fastapi", "httpx", "requests"]),
    ),
)

my_app = FastAPI()


@my_app.get("/")
def frontend():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>WebSocket Example</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 40px;
                background-color: #f2f2f2;
            }
            #messages {
                border: 1px solid #ccc;
                padding: 10px;
                max-width: 300px;
                margin-top: 20px;
            }
        </style>
    </head>
    <body>
        <h1>WebSocket Example</h1>
        <div id="messages"></div>
        <script>
            var socketUrl = "wss://apps.beam.cloud/4sjkx/something";
            var ws = new WebSocket(socketUrl);
            ws.onopen = function() {
                ws.send('{"Authorization": "Basic ODc4NmFkNjMyYWQ0YTBjZWIwMTA3M2Q1NTQyNjRjYmE6ZThhZWQxM2UzOWI1ZDFlZWExOTk4NTE2YmE0YjUyMDA="}');
            };
            ws.onmessage = function(event) {
                var messages = document.getElementById('messages');
                messages.innerHTML += '<p>' + event.data + '</p>';
            };
        </script>
    </body>
    </html>
    """
    return Response(content=html_content, media_type="text/html")


@my_app.websocket("/something")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    for i in range(50):
        await websocket.send_text(f"Message text was: {i}")
        await asyncio.sleep(0.5)


@app.asgi(workers=2, authorized=False)
def some_server(**inputs):
    return my_app
