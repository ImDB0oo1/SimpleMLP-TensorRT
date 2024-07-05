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

input = network.get_input(0)
output = network.get_output(0)
# Check if fast Half is avaliable
# print(builder.platform_has_fast_fp16)

profile = builder.create_optimization_profile()

min_shape = [1, 784]
opt_shape = [32, 784]
max_shape = [64, 784]
profile.set_shape(input.name, min_shape, opt_shape, max_shape)


config.add_optimization_profile(profile)
config.set_flag(trt.BuilderFlag.FP16)

# Build engine
engine_bytes = builder.build_serialized_network(network, config)

engine_path = "simple_mlp_dynamic.engine"
with open(engine_path, "wb") as f:
    f.write(engine_bytes)
