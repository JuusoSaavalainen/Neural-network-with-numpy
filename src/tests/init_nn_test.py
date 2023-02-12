import unittest
import numpy as np
from model.utility import init__nn, one_hot, normalize_zero_one, randomize_rows

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

class TestOneHot(unittest.TestCase):
    def test_one_hot(self):
        Y = 5
        expected = np.array([[0],[0],[0],[0],[0],[1],[0],[0],[0],[0]])
        result = one_hot(Y)
        self.assertTrue(np.allclose(result, expected))
            
    def test_one_hot_with_large_input(self):
        Y = 20
        with self.assertRaises(IndexError):
            one_hot(Y)
            
    def test_one_hot_with_float_input(self):
        Y = 5.5
        with self.assertRaises(IndexError):
            one_hot(Y)


class TestNormalizeZeroOne(unittest.TestCase):
    def test_normalize_zero_one(self):
        data = np.array([[255, 128, 64], [100, 200, 150]])
        expected = np.array([[1., 0.5, 0.25], [0.39215686274509803, 0.7843137254901961, 0.5882352941176471]])
        result = normalize_zero_one(data)
        np.testing.assert_almost_equal(result, expected, decimal=2)

class TestRandomizeRows(unittest.TestCase):
    def test_randomize_rows(self):
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = randomize_rows(data)
        self.assertEqual(len(result), len(data))
        self.assertFalse(np.array_equal(np.sort(result, axis=1), np.sort(data, axis=1)))
        
    def test_randomize_rows_same_result(self):
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result1 = randomize_rows(data)
        result2 = randomize_rows(data)
        self.assertFalse(np.array_equal(result1, result2))


if __name__ == '__main__':
    unittest.main()
