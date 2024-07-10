# SimpleMLP-TensorRT

A comprehensive guide and tutorial on using TensorRT for accelerating a simple Multi-Layer Perceptron (MLP). This repository includes step-by-step instructions, code examples, and explanations to help you get started with TensorRT for nueral network models.

**TensorRT Version**: Ensure you have TensorRT **version 8.6** or later installed, as TensorRT is compatible across versions from 8.6 onwards.

## Overview of Using TensorRT

1. **[Define Your Model](#Pytorch-model)**: Start by defining and training your model in PyTorch.
2. **[Convert to ONNX](#Convert-pytorch-model-to-ONNX)**: Convert your PyTorch model to the ONNX format. This step is necessary for both static and dynamic shape configurations.
3. **[Build TensorRT Engine](#Building-Engine)**:
   - **Static Shapes**: Build an engine with predefined input and output shapes for maximum optimization.
   - **Dynamic Shapes**: Build an engine with profile settings that support varying input and output shapes, allowing flexibility for different scenarios.
4. **[Inference from engine](#Inference)**:
   - **Create Execution Context**: Generate a context from the TensorRT engine to manage inference execution.
   - **Allocate Memory Buffers**:
     - Allocate memory for inputs and outputs in both host and device memory based on the shapes.
   - **Transfer Data and Run Inference**:
     - Transfer input data from the host to the device memory.
     - Execute inference using the TensorRT context.
     - Transfer the output data from the device back to the host memory.
   - **Post-Processing**: Reshape the 1D output array to the desired dimensions for further use.

By following these steps, you can leverage TensorRT to significantly improve the performance of your neural network models on NVIDIA GPUs.

## Static vs Dynamic Shapes in TensorRT

In TensorRT, the term "shapes" refers to the dimensions of the input and output tensors that a neural network processes.
When working with TensorRT, understanding the differences between static and dynamic shapes is crucial for optimizing and deploying models effectively.

**Note**: Working with different batch sizes required dynamic shapes.

### Static Shapes

Static shapes refer to input and output dimensions that are fixed and known at compile time. This means the shapes do not change during the inference process. Using static shapes allows TensorRT to perform optimizations that can lead to faster inference times because the engine can make assumptions about the tensor sizes.

#### Advantages of Static Shapes

- **Optimization**: TensorRT can apply more aggressive optimizations since the input dimensions are fixed.
- **Performance**: Inference can be faster due to the reduced overhead of managing variable tensor sizes.
- **Simplicity**: Easier to implement as there are no variations in input/output sizes to handle.

#### Disadvantages of Static Shapes

- **Flexibility**: Lack of flexibility to handle inputs of varying sizes without recompiling the engine.
- **Scalability**: Not suitable for applications where input dimensions change frequently.

### Dynamic Shapes

Dynamic shapes, on the other hand, refer to input and output dimensions that can vary at runtime. This flexibility allows a single TensorRT engine to handle inputs of different sizes, making it more versatile for applications that process data with varying dimensions.

#### Advantages of Dynamic Shapes

- **Flexibility**: Can handle inputs of varying sizes, making it suitable for a wider range of applications.
- **Scalability**: More scalable for different deployment scenarios where input sizes are not known in advance.

#### Disadvantages of Dynamic Shapes

- **Performance Overhead**: Potentially slower inference times due to the overhead of managing dynamic dimensions.
- **Complexity**: More complex to implement and manage, as the shapes need to be explicitly handled and specified during engine creation and inference.

### Choosing Between Static and Dynamic Shapes

The choice between static and dynamic shapes depends on the specific requirements of your application:
- Use **static shapes** if your input dimensions are fixed and known in advance, and you need maximum performance.
- Use **dynamic shapes** if your application needs to handle inputs of varying sizes and you require flexibility and scalability.

In the following sections, we will provide examples of how to implement both static and dynamic shapes using TensorRT for a simple MLP model.

## Pytorch model

First we define a simple Multi-Layer Perceptron (MLP) model using PyTorch. This model will be used as the basis for our TensorRT conversion and inference examples.
We need to train our model and save the weights,here we save them as (.pth) file.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Define the MLP model
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Parameters
input_size = 784  # Example for MNIST dataset (28x28 images)
hidden_size = 32
num_classes = 10
num_epochs = 10
batch_size = 1
learning_rate = 0.001

# Dummy dataset 
x_train = torch.randn(600, input_size)
y_train = torch.randint(0, num_classes, (600,))

train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Initialize the model, loss function, and optimizer
model = SimpleMLP(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# # Training loop
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

# # Save the trained model
torch.save(model.state_dict(), 'simple_mlp.pth')

print('Model training complete and saved to simple_mlp.pth')
```

## Convert pytorch model to ONNX

To leverage the power of TensorRT for both static and dynamic shapes, we first need to convert our PyTorch model to the ONNX (Open Neural Network Exchange) format. ONNX is an open format built to represent machine learning models, enabling them to be transferred between various frameworks and optimizers.

### Steps to Convert a PyTorch Model to ONNX

Regardless of whether you are working with static or dynamic shapes, the process of converting a PyTorch model to ONNX involves the following steps:

1. **Define Your Model**: Ensure your model is defined and trained in PyTorch.
2. **Create Dummy Input**: Prepare a dummy input tensor with the appropriate shape. For dynamic shapes, specify the axes that can vary.
3. **Export to ONNX**: Use the `torch.onnx.export` function to convert the model.

```python
import mlp
import torch


# Convert the model to ONNX format
# Dummy input for the model

# Parameters
input_size = 784  # Example for MNIST dataset (28x28 images)
hidden_size = 32
num_classes = 10
batch_size = 10


dummy_input = torch.randn(1, input_size)
onnx_file_path = "simple_mlp_dynamic.onnx"


model = mlp.SimpleMLP(input_size, hidden_size, num_classes)
# Load the model's weights (using the final epoch as an example)
model.load_state_dict(torch.load('simple_mlp.pth'))
model.eval()  # Set the model to evaluation mode

# Export the model
torch.onnx.export(
    model,
    dummy_input,
    onnx_file_path,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input' : {0: 'input_batch_size'}, # For static shapes we dont need dynamic axes
                  'output': {0: 'output_batch_size'}},
    opset_version=11
)

print(f'Model has been converted to ONNX and saved to {onnx_file_path}')
```

## Building Engine

After converting your PyTorch model to the ONNX format, the next step is to build a TensorRT engine. The engine is a highly optimized, platform-specific model that can run inference efficiently on NVIDIA GPUs.

**Note**: Engine should be build on your device because it optimize model based on your GPU architecture.

**Note**: Building the TensorRT engine can take a significant amount of time because it involves searching through various algorithms to optimize the inference performance.

### Steps to Build Engine

1. **Load the ONNX Model**: Read the ONNX file into memory.
2. **Create TensorRT Builder and Network**: Initialize the TensorRT builder and network.
3. **Parse the ONNX Model**: Parse the ONNX model to populate the TensorRT network.
4. **Build the Engine**: Configure the builder settings and build the engine.

For dynamic shapes, we need to set an optimization profile that includes minimum, optimal, and maximum shape values for the dynamic dimensions. During inference, the input shapes must not exceed the maximum or fall below the minimum values specified. The closer the input shapes are to the optimal values, the more performance benefits we can achieve with TensorRT.

```python
import tensorrt as trt

# Initialize TensorRT logger and builder
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(TRT_LOGGER)
config = builder.create_builder_config()


# Set cache
cache = config.create_timing_cache(b"")
config.set_timing_cache(cache, ignore_mismatch=False)


flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
builder.max_batch_size = 64
network = builder.create_network(flag)
parser = trt.OnnxParser(network, TRT_LOGGER)

path_onnx_model = "simple_mlp_dynamic.onnx"

with open(path_onnx_model, "rb") as f:
    if not parser.parse(f.read()):
        print(f"ERROR: Failed to parse the ONNX file {path_onnx_model}")
        for error in range(parser.num_errors):
            print(parser.get_error(error))

# Network has every inputs and its better to work with their names
# Since we only have 1 input we do this
input = network.get_input(0)

# Set profile for dynamic shapes(we dont need this step for static ones)
profile = builder.create_optimization_profile()
min_shape = [1, 784]
opt_shape = [32, 784]
max_shape = [64, 784]
profile.set_shape(input.name, min_shape, opt_shape, max_shape)
config.add_optimization_profile(profile)

# Check if fast Half is avaliable
# print(builder.platform_has_fast_fp16)
config.set_flag(trt.BuilderFlag.FP16)

# Build engine
engine_bytes = builder.build_serialized_network(network, config)

engine_path = "simple_mlp_dynamic.engine"
with open(engine_path, "wb") as f:
    f.write(engine_bytes)
```
## Inference

During inference, the following steps need to be performed:

1. **Create Context**: Generate a context from the pre-built TensorRT engine.
2. **Create CUDA Stream**: Create a `cuda.Stream()` to handle synchronization between host and device.
3. **Allocate Buffers**: Allocate memory buffers for inputs and outputs in both host and device memory exactly for how much data we need, based on the input and output shapes and size of data type we use.
5. **Transfer Input Data**: Move the input data from the host to the device memory.
6. **Run Inference**: Execute the inference on the device using the created context.
7. **Retrieve Output Data**: Transfer the output data from the device memory back to the host memory.

These steps ensure that the data flows correctly through the TensorRT engine for efficient inference and that synchronization between the host and device is properly managed.

### Static shapes

For static shapes, the input and output shapes are predefined and can be obtained from `engine.binding`. We simply need to allocate memory buffers based on these sizes.
```python
import numpy as np
import tensorrt as trt
from cuda import cuda, cudart
import ctypes
from typing import Optional, List

### Cudart keypoint handler
def check_cuda_err(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError("Cuda Error: {}".format(err))
    if isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError("Cuda Runtime Error: {}".format(err))
    else:
        raise RuntimeError("Unknown error type: {}".format(err))

def cuda_call(call):
    err, res = call[0], call[1:]
    check_cuda_err(err)
    if len(res) == 1:
        res = res[0]
    return res


### Class for transfer data between host and device memory 
class HostDeviceMem:
    """Pair of host and device memory, where the host memory is wrapped in a numpy array"""
    def __init__(self, size: int, dtype: np.dtype):
        nbytes = size * dtype.itemsize
        host_mem = cuda_call(cudart.cudaMallocHost(nbytes))
        pointer_type = ctypes.POINTER(np.ctypeslib.as_ctypes_type(dtype))

        self._host = np.ctypeslib.as_array(ctypes.cast(host_mem, pointer_type), (size,))
        self._device = cuda_call(cudart.cudaMalloc(nbytes))
        self._nbytes = nbytes

    @property
    def host(self) -> np.ndarray:
        return self._host

    @host.setter
    def host(self, arr: np.ndarray):
        if arr.size > self.host.size:
            raise ValueError(
                f"Tried to fit an array of size {arr.size} into host memory of size {self.host.size}"
            )
        #np.copyto(self.host[:arr.size], arr.flat, casting='safe')
        np.copyto(self.host[:arr.size], arr.flat)

    @property
    def device(self) -> int:
        return self._device

    @property
    def nbytes(self) -> int:
        return self._nbytes

    def __str__(self):
        return f"Host:\n{self.host}\nDevice:\n{self.device}\nSize:\n{self.nbytes}\n"

    def __repr__(self):
        return self.__str__()

    def free(self):
        cuda_call(cudart.cudaFree(self.device))
        cuda_call(cudart.cudaFreeHost(self.host.ctypes.data))


# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine: trt.ICudaEngine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda_call(cudart.cudaStreamCreate())
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))

        # Allocate host and device buffers
        bindingMemory = HostDeviceMem(size, dtype)

        # Append the device buffer to device bindings.
        bindings.append(int(bindingMemory.device))

        # Append to the appropriate list.
        if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
            inputs.append(bindingMemory)
        else:
            outputs.append(bindingMemory)


    return inputs, outputs, bindings, stream


# Frees the resources allocated in allocate_buffers
def free_buffers(inputs: List[HostDeviceMem], outputs: List[HostDeviceMem], stream: cudart.cudaStream_t):
    for mem in inputs + outputs:
        mem.free()
    cuda_call(cudart.cudaStreamDestroy(stream))


# Wrapper for cudaMemcpy which infers copy size and does error checking
def memcpy_host_to_device(device_ptr: int, host_arr: np.ndarray):
    nbytes = host_arr.size * host_arr.itemsize
    cuda_call(cudart.cudaMemcpy(device_ptr, host_arr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))


# Wrapper for cudaMemcpy which infers copy size and does error checking
def memcpy_device_to_host(host_arr: np.ndarray, device_ptr: int):
    nbytes = host_arr.size * host_arr.itemsize
    cuda_call(cudart.cudaMemcpy(host_arr, device_ptr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))


def _do_inference_base(inputs, outputs, stream, execute_async):
    # Transfer input data to the GPU.
    kind = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
    [cuda_call(cudart.cudaMemcpyAsync(inp.device, inp.host, inp.nbytes, kind, stream)) for inp in inputs]
    # Run inference.
    execute_async()
    # Transfer predictions back from the GPU.
    kind = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
    [cuda_call(cudart.cudaMemcpyAsync(out.host, out.device, out.nbytes, kind, stream)) for out in outputs]
    # Synchronize the stream
    cuda_call(cudart.cudaStreamSynchronize(stream))
    # Return only the host outputs.
    return [out.host for out in outputs]


# This function is generalized for multiple inputs/outputs for full dimension networks.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream):
    def execute_async():
        context.execute_async(bindings=bindings, stream_handle=stream)
    return _do_inference_base(inputs, outputs, stream, execute_async)

### Inference from tensorRT
engine_file_path = 'simple_mlp_dynamic.engine'
engine = load_engine(engine_file_path)

# Create execution context
context = engine.create_execution_context()

# Allocate buffers
inputs, outputs, binding, stream = allocate_buffers(engine)

# Dummy input data
input_data = np.random.randn(1, 784).astype(np.float32)

# Transfer input data to host memory
np.copyto(inputs[0].host, input_data.ravel())

# Run inference
output_data = do_inference(context, bindings, inputs, outputs, stream)
print("Inference output:", output_data)

# Free allocated memory
free_buffers(inputs, outputs, stream)
```
### Dynamic shapes

For dynamic shapes, we need to set the engine bindings based on the input shapes at inference time. Then, we allocate memory buffers accordingly based on these input shapes.

**Note**: We can define multiple profile settings with different minimum, optimal, and maximum shapes within a single TensorRT engine. This allows the engine to be used in various scenarios without needing to rebuild it, which can be time-consuming. For example, if we have two profile settings and three bindings for input and output, to access the first input of the second profile, you would use `get_binding(3 (number of bindings) + 0 (first input))`.

**Note**: The output from TensorRT is a 1D array. We need to reshape this array to the desired dimensions to use it as the output of our model.

```python
import numpy as np
import tensorrt as trt
from cuda import cuda, cudart
import ctypes
from typing import Optional, List


### Cudart keypoint handler
def check_cuda_err(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError("Cuda Error: {}".format(err))
    if isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError("Cuda Runtime Error: {}".format(err))
    else:
        raise RuntimeError("Unknown error type: {}".format(err))

def cuda_call(call):
    err, res = call[0], call[1:]
    check_cuda_err(err)
    if len(res) == 1:
        res = res[0]
    return res


### Class for transfer data between host and device memory 
class HostDeviceMem:
    """Pair of host and device memory, where the host memory is wrapped in a numpy array"""
    def __init__(self, size: int, dtype: np.dtype):
        nbytes = size * dtype.itemsize
        host_mem = cuda_call(cudart.cudaMallocHost(nbytes))
        pointer_type = ctypes.POINTER(np.ctypeslib.as_ctypes_type(dtype))

        self._host = np.ctypeslib.as_array(ctypes.cast(host_mem, pointer_type), (size,))
        self._device = cuda_call(cudart.cudaMalloc(nbytes))
        self._nbytes = nbytes

    @property
    def host(self) -> np.ndarray:
        return self._host

    @host.setter
    def host(self, arr: np.ndarray):
        if arr.size > self.host.size:
            raise ValueError(
                f"Tried to fit an array of size {arr.size} into host memory of size {self.host.size}"
            )
        #np.copyto(self.host[:arr.size], arr.flat, casting='safe')
        np.copyto(self.host[:arr.size], arr.flat)

    @property
    def device(self) -> int:
        return self._device

    @property
    def nbytes(self) -> int:
        return self._nbytes

    def __str__(self):
        return f"Host:\n{self.host}\nDevice:\n{self.device}\nSize:\n{self.nbytes}\n"

    def __repr__(self):
        return self.__str__()

    def free(self):
        cuda_call(cudart.cudaFree(self.device))
        cuda_call(cudart.cudaFreeHost(self.host.ctypes.data))


# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
# If engine uses dynamic shapes, specify a profile to find the maximum input & output size.
def allocate_buffers(engine: trt.ICudaEngine, inputs_shape):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda_call(cudart.cudaStreamCreate())
    tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    for shape, binding in zip(inputs_shape, tensor_names):
        size = trt.volume(shape)
        dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(binding)))

        # Allocate host and device buffers
        bindingMemory = HostDeviceMem(size, dtype)

        # Append the device buffer to device bindings.
        bindings.append(int(bindingMemory.device))

        # Append to the appropriate list.
        if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
            inputs.append(bindingMemory)
        else:
            outputs.append(bindingMemory)


    return inputs, outputs, bindings, stream


# Frees the resources allocated in allocate_buffers
def free_buffers(inputs: List[HostDeviceMem], outputs: List[HostDeviceMem], stream: cudart.cudaStream_t):
    for mem in inputs + outputs:
        mem.free()
    cuda_call(cudart.cudaStreamDestroy(stream))


# Wrapper for cudaMemcpy which infers copy size and does error checking
def memcpy_host_to_device(device_ptr: int, host_arr: np.ndarray):
    nbytes = host_arr.size * host_arr.itemsize
    cuda_call(cudart.cudaMemcpy(device_ptr, host_arr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))


# Wrapper for cudaMemcpy which infers copy size and does error checking
def memcpy_device_to_host(host_arr: np.ndarray, device_ptr: int):
    nbytes = host_arr.size * host_arr.itemsize
    cuda_call(cudart.cudaMemcpy(host_arr, device_ptr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))


def _do_inference_base(inputs, outputs, stream, execute_async):
    # Transfer input data to the GPU.
    kind = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
    [cuda_call(cudart.cudaMemcpyAsync(inp.device, inp.host, inp.nbytes, kind, stream)) for inp in inputs]
    # Run inference.
    execute_async()
    # Transfer predictions back from the GPU.
    kind = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
    [cuda_call(cudart.cudaMemcpyAsync(out.host, out.device, out.nbytes, kind, stream)) for out in outputs]
    # Synchronize the stream
    cuda_call(cudart.cudaStreamSynchronize(stream))
    # Return only the host outputs.
    return [out.host for out in outputs]


# This function is generalized for multiple inputs/outputs for full dimension networks.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference_v2(context, bindings, inputs, outputs, stream):
    def execute_async():
        context.execute_async_v2(bindings=bindings, stream_handle=stream)
    return _do_inference_base(inputs, outputs, stream, execute_async)


### Inference from tensorRt
# Function to load a TensorRT engine from a file
def load_engine(engine_file_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

# Load the engine
engine_file_path = "/home/jetson/Danial/a/superpoint.engine"
engine = load_engine(engine_file_path)


# Create context
context = engine.create_execution_context()
# In dynamic version we need to first load input and set binding based on input shapes then allocate buffers
# Dummy input data
input_data = np.random.randn(40, 784).astype(np.float32)

# Set binding
input_name = engine.get_tensor_name(0)
context.set_input_shape(input_name, input_data.shape)

# Set input shapes for memory allocation
# We should know what model outputs shapes
model_shapes = [input_data.shape, (input_data.shape[0], 10)] #[input_shape=(40, 784), output_shape=(40, 10)]

# Allocate memory for inputs and outputs
inputs, outputs, bindings, stream = allocate_buffers(engine, inputs_shape)

# Transfer input data to the allocated buffer
np.copyto(inputs[0].host, img.ravel())

output_data = do_inference_v2(context, bindings, inputs, outputs, stream)

# Reshape the output to desire shape and convert to torch 
output = torch.from_numpy(output_data[0].reshape(input_data.shape[0], 10))

# Free allocated memory
free_buffers(inputs, outputs, stream)
```
