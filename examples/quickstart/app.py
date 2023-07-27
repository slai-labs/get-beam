"""
Run this on Beam:

beam run app.py:multiply_numbers
"""

from beam import App, Runtime

app = App(name="hello-beam", runtime=Runtime())


@app.run()
def multiply_numbers():
    print("This is running remotely on Beam!")
    x = 43
    y = 177
    print(f"🔮 {x} * {y} is { x * y}")
