"""Unittests for ElasticDL's Tensor data structure."""
import unittest

import numpy as np

from elasticdl.python.common.tensor import ndarray_to_pb, pb_to_ndarry


class TensorTest(unittest.TestCase):
    def test_round_trip(self):
        def verify(array):
            pb = ndarray_to_pb(array)
            new_array = pb_to_ndarry(pb)
            np.testing.assert_array_equal(array, new_array)

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


if __name__ == "__main__":
    unittest.main()
