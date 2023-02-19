import unittest
import numpy as np
import model.dataformat as dataformat
from model.utility import NeuralNetwork

class TestOneHot(unittest.TestCase):
    def setUp(self):
        self.nn = NeuralNetwork([784,10,10])

    def test_one_hot(self):
        Y = 5
        expected = np.array([[0],[0],[0],[0],[0],[1],[0],[0],[0],[0]])
        result = self.nn.one_hot(Y)
        self.assertTrue(np.allclose(result, expected))
            
    def test_one_hot_with_large_input(self):
        Y = 20
        with self.assertRaises(IndexError):
            self.nn.one_hot(Y)
            
    def test_one_hot_with_float_input(self):
        Y = 5.5
        with self.assertRaises(IndexError):
            self.nn.one_hot(Y)


class TestNormalizeZeroOne(unittest.TestCase):
    def test_normalize_zero_one(self):
        data = np.array([[255, 128, 64], [100, 200, 150]])
        expected = np.array([[1., 0.5, 0.25], [0.39215686274509803, 0.7843137254901961, 0.5882352941176471]])
        result = dataformat.normalize_zero_one(data)
        np.testing.assert_almost_equal(result, expected, decimal=2)

class TestRandomizeRows(unittest.TestCase):
    def test_randomize_rows(self):
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = dataformat.randomize_rows(data)
        self.assertEqual(len(result), len(data))
        self.assertTrue(np.array_equal(np.sort(result, axis=1), np.sort(data, axis=1)))
        
    def test_randomize_rows_same_result(self):
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result1 = dataformat.randomize_rows(data)
        result2 = dataformat.randomize_rows(data)
        self.assertFalse(np.array_equal(result1, result2))


if __name__ == '__main__':
    unittest.main()
