# SimpleMLP-TensorRT

A comprehensive guide and tutorial on using TensorRT for accelerating a simple Multi-Layer Perceptron (MLP). This repository includes step-by-step instructions, code examples, and explanations to help you get started with TensorRT for nueral network models.

## Overview of Using TensorRT

1. **Define Your Model**: Start by defining and training your model in PyTorch.
      - See [Pytorch model](#Pytorch-model)
3. **Convert to ONNX**: Convert your PyTorch model to the ONNX format. This step is necessary for both static and dynamic shape configurations.
4. **Build TensorRT Engine**:
   - **Static Shapes**: Build an engine with predefined input and output shapes for maximum optimization.
   - **Dynamic Shapes**: Build an engine with profile settings that support varying input and output shapes, allowing flexibility for different scenarios.
5. **Inference from engine**:
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

## Convert pytorch model to ONNX

To leverage the power of TensorRT for both static and dynamic shapes, we first need to convert our PyTorch model to the ONNX (Open Neural Network Exchange) format. ONNX is an open format built to represent machine learning models, enabling them to be transferred between various frameworks and optimizers.

### Steps to Convert a PyTorch Model to ONNX

Regardless of whether you are working with static or dynamic shapes, the process of converting a PyTorch model to ONNX involves the following steps:

1. **Define Your Model**: Ensure your model is defined and trained in PyTorch.
2. **Create Dummy Input**: Prepare a dummy input tensor with the appropriate shape. For dynamic shapes, specify the axes that can vary.
3. **Export to ONNX**: Use the `torch.onnx.export` function to convert the model.

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

### Dynamic shapes

For dynamic shapes, we need to set the engine bindings based on the input shapes at inference time. Then, we allocate memory buffers accordingly based on these input shapes.

**Note**: We can define multiple profile settings with different minimum, optimal, and maximum shapes within a single TensorRT engine. This allows the engine to be used in various scenarios without needing to rebuild it, which can be time-consuming. For example, if we have two profile settings and three bindings for input and output, to access the first input of the second profile, you would use `get_binding(3 (number of bindings) + 0 (first input))`.

**Note**: The output from TensorRT is a 1D array. We need to reshape this array to the desired dimensions to use it as the output of our model.
