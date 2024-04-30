# from beam import (
#     App,
#     Runtime,
#     Image,
#     Volume,
#     QueueDepthAutoscaler,
#     PythonVersion,
#     GpuType,
# )
# import torch 
# from transformers import LlamaTokenizer, LlamaForCausalLM

# # Beam Volume to store cached models
# CACHE_PATH = "./cached_models"

# app = App(
#     name="high-performance-inference",
#     runtime=Runtime(
#         cpu=4,
#         memory="32Gi",
#         gpu=GpuType.A10G,
#         image=Image(
#             python_version=PythonVersion.Python310,
#             python_packages=[
#                 "accelerate",
#                 "sentencepiece",
#                 "transformers",
#                 "torch",
#             ],  # You can also add a path to a requirements.txt instead
#         ),
#     ),
#     # Storage Volume for model weights
#     volumes=[Volume(name="cached_models", path=CACHE_PATH)],
# )

# # Autoscale by queue depth
# autoscaler = QueueDepthAutoscaler(max_tasks_per_replica=5, max_replicas=2)


# # This function runs once when the container boots
# def load_models():
#     model_path = 'openlm-research/open_llama_7b'
    
#     model = LlamaForCausalLM.from_pretrained(
#         model_path, torch_dtype=torch.float16, device_map='auto',
#     )
#     tokenizer = LlamaTokenizer.from_pretrained(model_path)


#     return model, tokenizer


# # Rest API initialized with loader and autoscaler
# @app.rest_api(loader=load_models, autoscaler=autoscaler, authorized=False)
# def predict(**inputs):
#     # Retrieve cached model from loader
#     model, tokenizer = inputs["context"]

#     try:
#         prompt = inputs["prompt"]
#     # Use a default prompt if none is provided
#     except KeyError:
#         prompt = "Q: What is the largest animal?\nA:"

#     input_ids = tokenizer(prompt, return_tensors="pt").to('cuda').input_ids

#     generation_output = model.generate(
#         input_ids=input_ids, max_new_tokens=32
#     )

#     result = tokenizer.decode(generation_output[0])
#     print(result)

#     return {"prediction": result}