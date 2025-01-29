from mlp_trt import SimpleMLPTRT

import torch
import random

# Model parameters
weights_file_path = 'weights/simple_mlp.pth'
onnx_file_path = 'weights/simple_mlp.onnx'
engine_file_path = 'weights/simple_mlp.engine'
input_size = 784
hidden_size = 32
num_classes = 10


# Create an instance of the inference class
trt_model = SimpleMLPTRT(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes, weights_file_path=weights_file_path, onnx_file_path=onnx_file_path, engine_file_path=engine_file_path)

# Inference with dynamic batch_size
for i in range(100):
    # Create dummy input with dynamic batch_size
    batch_size = random.randint(1, 60)  
    dummy_input = torch.randn(batch_size, 784)

    # Perform inference
    output = trt_model.infer(dummy_input)
