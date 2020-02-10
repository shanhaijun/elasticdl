import numpy as np

from elasticdl.proto import elasticdl_pb2
from elasticdl.python.common.dtypes import (
    dtype_numpy_to_tensor,
    dtype_tensor_to_numpy,
)


def ndarray_to_pb(array):
    pb = elasticdl_pb2.Tensor()
    dtype = dtype_numpy_to_tensor(array.dtype)
    if not dtype:
        raise ValueError("Dtype of ndarray %s is not supported", array.dtype)
    pb.dtype = dtype
    pb.dims.extend(array.shape)
    pb.content = array.tobytes()
    return pb


def pb_to_ndarry(pb):
    if not pb.dims:
        raise ValueError("PB has no dim defined")
    dtype = dtype_tensor_to_numpy(pb.dtype)
    # Check that the buffer size agrees with dimensions.
    size = dtype.itemsize
    for d in pb.dims:
        size *= d
    if size != len(pb.content):
        raise ValueError(
            "PB size mismatch, dim: %s, len(content): %d",
            pb.dims,
            len(pb.content),
        )
    array = np.ndarray(shape=pb.dims, dtype=dtype, buffer=pb.content)
    return array
