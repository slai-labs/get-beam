'''
Gradio App on Beam

Serve it: beam serve app.py
Deploy it: beam deploy app.py
'''

import subprocess
import time

import httpx
from beam import App, Image, Runtime
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.responses import Response as StarletteResponse

runtime = Runtime(
    cpu=1,
    memory="2Gi",
    image=Image(
        python_version="python3.8",
        python_packages=["gradio", "fastapi", "uvicorn", "requests", "httpx"],
    ),
)
app = App(name="gradio-app", runtime=runtime)


GRADIO_URL = "http://127.0.0.1:7860"  # Assuming default Gradio port.


# Starts the Gradio server
def start_frontend():
    subprocess.Popen(["python3", "frontend.py"])
    start_time = time.time()
    elapsed = 0
    # Wait for Gradio server to be up.
    while elapsed < 60:
        try:
            with httpx.Client() as client:
                response = client.get(GRADIO_URL)
                if response.status_code == 200:
                    print("Gradio app is ready")
                    break
        except httpx.RequestError:
            print("Gradio server is not up yet, waiting...")
            time.sleep(1)
            elapsed = time.time() - start_time


# FastAPI Entry Point
@app.asgi(loader=start_frontend, authorized=False)
def fastapi_app():
    asgi_app = FastAPI()

    @asgi_app.route("/health")
    async def health(request: Request):
        return JSONResponse({"OK": True})

    @asgi_app.route("/{path:path}", include_in_schema=False, methods=["GET", "POST"])
    async def proxy_all(request: Request):
        """
        Proxy all requests to Gradio.
        """
        path = request.url.path.lstrip("/")  # strip leading slash
        headers = dict(request.headers)

        async with httpx.AsyncClient() as client:
            response = await client.request(
                request.method,
                f"{GRADIO_URL}/{path}",
                headers=headers,
                data=await request.body(),
                params=request.query_params,
            )

            content = await response.aread()
            response_headers = dict(response.headers)
            return StarletteResponse(
                content=content,
                status_code=response.status_code,
                headers=response_headers,
            )

    return asgi_app