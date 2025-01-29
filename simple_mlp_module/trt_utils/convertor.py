import tensorrt as trt
import torch


# Convert the model to ONNX format
# Dummy input for the model (replace with an actual sample if needed)
def build_onnx(weights_file_path, onnx_file_path, input_size, model):
    
    dummy_input = torch.randn(1, input_size)
    
    # Load the model's weights (using the final epoch as an example)
    model.load_state_dict(torch.load(weights_file_path))
    model.eval()  # Set the model to evaluation mode

    # Export the model
    torch.onnx.export(
        model,
        dummy_input,
        onnx_file_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input' : {0: 'input_batch_size'},
                    'output': {0: 'output_batch_size'}},
        opset_version=11
    )

    print(f'Model has been converted to ONNX and saved to {onnx_file_path}')
def build_engine(onnx_path, engine_path):
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


    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print(f"ERROR: Failed to parse the ONNX file {onnx_path}")
            for error in range(parser.num_errors):
                print(parser.get_error(error))

    input = network.get_input(0)
    # Check if fast Half is avaliable
    # print(builder.platform_has_fast_fp16)

    profile = builder.create_optimization_profile()

    min_shape = [1, 784]
    opt_shape = [32, 784]
    max_shape = [64, 784]
    profile.set_shape(input.name, min_shape, opt_shape, max_shape)


    config.add_optimization_profile(profile)
    #config.set_flag(trt.BuilderFlag.FP16)

    # Build engine
    engine_bytes = builder.build_serialized_network(network, config)

    with open(engine_path, "wb") as f:
        f.write(engine_bytes)

def load_engine(engine_file_path: str):
    TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())