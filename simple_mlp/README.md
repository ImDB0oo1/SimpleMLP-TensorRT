This part provides an example of a simple MLP model, including its conversion to ONNX, optimization into a TensorRT engine, and inference using the TensorRT context. 

To use this with your own model, you will need to modify the MLP model, `build_onnx`, `build_engine` in `convertor.py`, and the inference part in the `SimpleMLPTRT` class to accommodate the input and output dimensions of your model. After making these adjustments, simply use `main.py` as shown in the example.
