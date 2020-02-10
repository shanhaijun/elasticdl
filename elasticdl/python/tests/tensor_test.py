"""Unittests for ElasticDL's Tensor data structure."""
import unittest

import numpy as np

from elasticdl.proto import elasticdl_pb2
from elasticdl.python.common.dtypes import dtype_numpy_to_tensor
from elasticdl.python.common.tensor import (
    Tensor,
    deserialize_tensor_pb,
    serialize_tensor,
    tensor_pb_to_ndarray,
    tensor_pb_to_tf_tensor,
)


class TensorTest(unittest.TestCase):
    def test_tensor_data_structure(self):
        arr = np.ndarray(shape=[3, 1, 2, 4], dtype=np.int32)
        tensor = Tensor(arr)
        self.assertTrue(np.array_equal(arr, tensor.values))
        self.assertTrue(np.array_equal(arr, tensor.to_tf_tensor()))

        # Test round trip
        # tensor to tensor PB
        tensor = Tensor(arr)
        pb = tensor.to_tensor_pb()
        self.assertEqual(pb.dims, [3, 1, 2, 4])
        self.assertEqual(pb.dtype, elasticdl_pb2.DT_INT32)

        # tensor PB to tensor
        tensor_new = Tensor.from_tensor_pb(pb)
        np.testing.assert_array_equal(tensor_new.values, arr)

        # Test Tensor().to_ndarray()
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        tensor = Tensor(values)
        self.assertTrue(np.allclose(values, tensor.to_ndarray()))

    def test_serialize_tensor(self):
        def _ndarray_to_tensor_pb(values):
            return Tensor(values).to_tensor_pb()

        # Wrong type, should raise
        arr = np.array([1, 2, 3, 4], dtype=np.uint8)
        with self.assertRaises(ValueError):
            _ndarray_to_tensor_pb(arr)

        # Empty array
        arr = np.array([], dtype=np.float32)
        t = _ndarray_to_tensor_pb(arr)
        self.assertEqual([0], t.dims)
        self.assertEqual(0, len(t.content))

        # Pathological case, one of the dimensions is 0.
        arr = np.ndarray(shape=[2, 0, 1, 9], dtype=np.float32)
        t = _ndarray_to_tensor_pb(arr)
        self.assertEqual([2, 0, 1, 9], t.dims)
        self.assertEqual(0, len(t.content))

        # 1-D array
        arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        t = _ndarray_to_tensor_pb(arr)
        self.assertEqual([4], t.dims)
        self.assertEqual(4 * 4, len(t.content))

        # 4-D random array
        arr = np.ndarray(shape=[2, 1, 3, 4], dtype=np.float32)
        t = _ndarray_to_tensor_pb(arr)
        self.assertEqual([2, 1, 3, 4], t.dims)
        self.assertEqual(4 * 2 * 1 * 3 * 4, len(t.content))

        # test name argument
        arr = np.ndarray(shape=[2, 1, 3, 4], dtype=np.float32)
        t = _ndarray_to_tensor_pb(arr)
        self.assertEqual([2, 1, 3, 4], t.dims)
        self.assertEqual(4 * 2 * 1 * 3 * 4, len(t.content))

    def test_deserialize_tensor_pb(self):
        pb = elasticdl_pb2.Tensor()
        tensor = Tensor()
        # No dim defined, should raise.
        self.assertRaises(ValueError, deserialize_tensor_pb, pb, tensor)

        # Empty array, should be ok.
        pb.dims.append(0)
        pb.content = b""
        pb.dtype = elasticdl_pb2.DT_FLOAT32
        deserialize_tensor_pb(pb, tensor)
        np.testing.assert_array_equal(
            np.array([], dtype=np.float32), tensor.values
        )

        # Wrong type, should raise
        del pb.dims[:]
        pb.dims.append(0)
        pb.content = b""
        pb.dtype = elasticdl_pb2.DT_INVALID
        self.assertRaises(ValueError, deserialize_tensor_pb, pb, tensor)

        # Pathological case, one of the dimensions is 0.
        del pb.dims[:]
        pb.dims.extend([2, 0, 1, 9])
        pb.content = b""
        pb.dtype = elasticdl_pb2.DT_FLOAT32
        deserialize_tensor_pb(pb, tensor)
        np.testing.assert_array_equal(
            np.ndarray(shape=[2, 0, 1, 9], dtype=np.float32), tensor.values
        )

        # Wrong content size, should raise
        del pb.dims[:]
        pb.dims.append(11)
        pb.content = b"\0" * (4 * 12)
        pb.dtype = elasticdl_pb2.DT_FLOAT32
        self.assertRaises(ValueError, deserialize_tensor_pb, pb, tensor)

        # Compatible dimensions, should be ok.
        for m in (1, 2, 3, 4, 6, 12):
            del pb.dims[:]
            pb.content = b"\0" * (4 * 12)
            pb.dims.extend([m, 12 // m])
            pb.dtype = elasticdl_pb2.DT_FLOAT32
            deserialize_tensor_pb(pb, tensor)
            self.assertEqual((m, 12 // m), tensor.values.shape)
            self.assertTrue(isinstance(tensor.values, np.ndarray))

    def test_round_trip(self):
        def verify(values):
            tensor = Tensor(values)
            pb = elasticdl_pb2.Tensor()
            serialize_tensor(tensor, pb)
            tensor_new = Tensor()
            deserialize_tensor_pb(pb, tensor_new)
            np.testing.assert_array_equal(values, tensor_new.values)

        # dtype = np.float32
        # 1-D array
        verify(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        # 4-D random array
        verify(np.ndarray(shape=[2, 1, 3, 4], dtype=np.float32))

        # dtype = np.int64
        # 1-D random array
        verify(np.array([1, 2, 3, 4], dtype=np.int64))
        # 4-D random array
        verify(np.ndarray(shape=[2, 1, 3, 4], dtype=np.int64))

    def _create_tensor_pb(self, values):
        pb = elasticdl_pb2.Tensor()
        pb.dims.extend(values.shape)
        pb.dtype = dtype_numpy_to_tensor(values.dtype)
        pb.content = values.tobytes()
        return pb

    def test_tensor_pb_to_ndarray(self):
        values = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], np.float32)
        pb = self._create_tensor_pb(values)
        self.assertTrue(np.allclose(tensor_pb_to_ndarray(pb), values))

    def test_tensor_pb_to_tf_tensor(self):
        values = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], np.float32)

        # Test dense tensor
        pb = self._create_tensor_pb(values)
        self.assertTrue(np.allclose(tensor_pb_to_tf_tensor(pb), values))


if __name__ == "__main__":
    unittest.main()
