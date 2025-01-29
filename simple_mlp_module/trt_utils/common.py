import os
import numpy as np
import tensorrt as trt
from cuda import cuda, cudart
import ctypes
from typing import List, Tuple


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


class HostDeviceMem:
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
        np.copyto(self.host[: arr.size], arr.flat)

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


def allocate_buffers(engine: trt.ICudaEngine, inputs_shape: List[Tuple[int]]):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda_call(cudart.cudaStreamCreate())
    tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    for shape, binding in zip(inputs_shape, tensor_names):
        size = trt.volume(shape)
        dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(binding)))

        bindingMemory = HostDeviceMem(size, dtype)
        bindings.append(int(bindingMemory.device))

        if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
            inputs.append(bindingMemory)
        else:
            outputs.append(bindingMemory)

    return inputs, outputs, bindings, stream


def free_buffers(
    inputs: List[HostDeviceMem],
    outputs: List[HostDeviceMem],
    stream: cudart.cudaStream_t,
):
    for mem in inputs + outputs:
        mem.free()
    cuda_call(cudart.cudaStreamDestroy(stream))


def memcpy_host_to_device(device_ptr: int, host_arr: np.ndarray):
    nbytes = host_arr.size * host_arr.itemsize
    cuda_call(
        cudart.cudaMemcpy(
            device_ptr, host_arr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        )
    )


def memcpy_device_to_host(host_arr: np.ndarray, device_ptr: int):
    nbytes = host_arr.size * host_arr.itemsize
    cuda_call(
        cudart.cudaMemcpy(
            host_arr, device_ptr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
        )
    )


def _do_inference_base(inputs, outputs, stream, execute_async):
    kind = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
    [
        cuda_call(
            cudart.cudaMemcpyAsync(inp.device, inp.host, inp.nbytes, kind, stream)
        )
        for inp in inputs
    ]
    execute_async()
    kind = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
    [
        cuda_call(
            cudart.cudaMemcpyAsync(out.host, out.device, out.nbytes, kind, stream)
        )
        for out in outputs
    ]
    cuda_call(cudart.cudaStreamSynchronize(stream))
    return [out.host for out in outputs]


def do_inference_v2(context, bindings, inputs, outputs, stream):
    def execute_async():
        context.execute_async_v2(bindings=bindings, stream_handle=stream)

    return _do_inference_base(inputs, outputs, stream, execute_async)
