import unittest
import numpy as np
from model.utility import init__nn

class TestNNInit(unittest.TestCase):
    def test_init__nn(self):
        layers_dims = [2, 3, 1]
        params = init__nn(layers_dims)
        self.assertEqual(len(params), 2 * (len(layers_dims) - 1))
        for i in range(1, len(layers_dims)):
            W = params[f'W{i}']
            b = params[f'b{i}']
            self.assertEqual(W.shape, (layers_dims[i], layers_dims[i-1]))
            self.assertEqual(b.shape, (layers_dims[i], 1))

    def test_params_not_nan(self):
        layers_dims = [2, 3, 1]
        params = init__nn(layers_dims)
        for i in range(1, len(layers_dims)):
            W = params[f'W{i}']
            b = params[f'b{i}']
            self.assertFalse(np.any(np.isnan(W)))
            self.assertFalse(np.any(np.isnan(b)))
