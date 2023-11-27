import random
import gradio as gr


def random_response(message, history):
    return random.choice(["Yes", "No"])


demo = gr.ChatInterface(random_response)

demo.launch()