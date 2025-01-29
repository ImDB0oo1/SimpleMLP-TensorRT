import os
import torch
import numpy as np
from trt_utils.convertor import build_onnx, build_engine, load_engine
import trt_utils.common as common
import mlp

class SimpleMLPTRT:
    def __init__(self, input_size=784, hidden_size=32, num_classes=10, batch_size=1, weights_file_path='simple_mlp.pth', onnx_file_path='simple_mlp.onnx', engine_file_path='simple_mlp.engine'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.batch_size = batch_size
        
        # Define model (not used directly for inference but can be useful for validation)
        self.model = mlp.SimpleMLP(input_size, hidden_size, num_classes)

        # File paths
        self.weights_file_path = weights_file_path
        self.onnx_file_path = onnx_file_path
        self.engine_file_path = engine_file_path

        # Build ONNX model if it doesn't exist
        if not os.path.exists(self.onnx_file_path):
            build_onnx(self.weights_file_path, self.onnx_file_path, input_size, self.model)
            print(f'ONNX model saved to {self.onnx_file_path}')
        else:
            print('ONNX file already exists!')

        # Build TensorRT engine if it doesn't exist
        if not os.path.exists(self.engine_file_path):
            build_engine(self.onnx_file_path, self.engine_file_path)
            print(f'TensorRT engine saved to {self.engine_file_path}')
        else:
            print('TensorRT engine file already exists!')

        # Load TensorRT engine
        self.engine = load_engine(self.engine_file_path)
        
        # Create execution context for inference
        self.context = self.engine.create_execution_context()

    def infer(self, input_tensor):
        """Perform inference on the input tensor."""

        # Set binding for context based on the input shape
        input_shape = input_tensor.shape
        output_shape = (input_shape[0], self.num_classes)
        
        input_binding_index = self.engine.get_tensor_name(0)
        self.context.set_input_shape(input_binding_index, input_shape)

        # Allocate buffers for inputs and outputs
        model_shapes = [input_shape, output_shape]
        inputs, outputs, bindings, stream = common.allocate_buffers(self.engine, model_shapes)

        # Transfer data to host memory
        np.copyto(inputs[0].host, input_tensor.ravel())

        # Do inference
        common.do_inference_v2(self.context, bindings, inputs, outputs, stream)

        # Post-process the output of the model
        output = torch.from_numpy(outputs[0].host.reshape(output_shape).copy())
        
        # Free buffers
        common.free_buffers(inputs, outputs, stream)

        return output


