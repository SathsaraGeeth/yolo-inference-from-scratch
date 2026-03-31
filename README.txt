In this repository, we reimplement the YOLOv5n model without using any deep learning libraries. This allows it to run on custom hardware or embedded devices that do not support such libraries, while also removing their overhead.

The project is divided into several parts:

1. Model extraction  
model_extract.ipynb – Extracts model parameters, weights, and structure into a canonical format. Running this notebook generates the output in the model/ directory, as below.

atomic_dag.pdf
classes.npy
config.npy
graph.npy
weights.npy

2. Inference testing  
inference_testing.ipynb – Runs the model using the extracted canonical format, still using the PyTorch framework.

3. Inference  
inference_python.ipynb – Implements the required PyTorch functionalities in C (ongoing).
