import numpy as np
import tensorflow as tf

from elasticdl.proto import elasticdl_pb2
from elasticdl.python.common.dtypes import (
    dtype_numpy_to_tensor,
    dtype_tensor_to_numpy,
)


class Tensor(object):
    """Data structure for tensors in ElasticDL.
    `Tensor` represents a dense tensor.
    """

    def __init__(self, values=None):
        self.set(values)

    @classmethod
    def from_tensor_pb(cls, tensor_pb):
        """Create an ElasticDL Tensor object from tensor protocol buffer.

        Return the created Tensor object.
        """
        tensor = cls()
        deserialize_tensor_pb(tensor_pb, tensor)
        return tensor

    def set(self, values=None):
        self.values = (
            values.numpy() if isinstance(values, tf.Tensor) else values
        )

    def to_tensor_pb(self):
        tensor_pb = elasticdl_pb2.Tensor()
        serialize_tensor(self, tensor_pb)
        return tensor_pb

    def to_tf_tensor(self):
        return tf.constant(self.values)

    def to_ndarray(self):
        return self.values


def serialize_tensor(tensor, tensor_pb):
    """Serialize ElasticDL Tensor to tensor protocol buffer."""
    dtype = dtype_numpy_to_tensor(tensor.values.dtype)
    if not dtype:
        raise ValueError(
            "Dtype of ndarray %s is not supported", tensor.values.dtype
        )
    tensor_pb.dtype = dtype
    tensor_pb.dims.extend(tensor.values.shape)
    tensor_pb.content = tensor.values.tobytes()


def deserialize_tensor_pb(tensor_pb, tensor):
    """Deserialize tensor protocol buffer to ElasticDL Tensor.

    Note that the input tensor protocol buffer is reset and underlying buffer
    is passed to the returned ndarray.
    """
    if not tensor_pb.dims:
        raise ValueError("Tensor PB has no dim defined")

    dtype = dtype_tensor_to_numpy(tensor_pb.dtype)
    # Check that the buffer size agrees with dimensions.
    size = dtype.itemsize
    for d in tensor_pb.dims:
        size *= d
    if size != len(tensor_pb.content):
        raise ValueError(
            "Tensor PB size mismatch, dim: %s, len(content): %d",
            tensor_pb.dim,
            len(tensor_pb.content),
        )
    tensor.set(
        values=np.ndarray(
            shape=tensor_pb.dims, dtype=dtype, buffer=tensor_pb.content
        ),
    )
    tensor_pb.Clear()


def tensor_pb_to_ndarray(tensor_pb):
    """Deserialize tensor protocol buffer and return a numpy ndarray."""
    return Tensor.from_tensor_pb(tensor_pb).to_ndarray()


def tensor_pb_to_tf_tensor(tensor_pb):
    """Deserialize tensor protocol buffer and return a TensorFlow tensor."""
    return Tensor.from_tensor_pb(tensor_pb).to_tf_tensor()


def emplace_tensor_pb_from_ndarray(
    tensor_pb_list, values, indices=None, name=None
):
    """Generate a tensor procotol buffer and append it to tensor_pb_list.

    Note:
        This function does not use list append function as following code
            snippet. It is slow because append function will copy the input
            protocol buffer.

        ```
        pb = elasticdl_pb2.Tensor()
        pb.dims.extend([3])
        pb.dtype = DT_INT64
        pb.content = np.array([1, 2, 3]).tobytes()
        tensor_pb_list.append(pb) # slow, because append copies pb
        ```
    """
    tensor_pb = tensor_pb_list.add()
    tensor = Tensor(values)
    serialize_tensor(tensor, tensor_pb)
