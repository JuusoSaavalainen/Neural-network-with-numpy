import unittest
import numpy as np
from model.utility import NeuralNetwork


class TestActivationFunctions(unittest.TestCase):
    def setUp(self):
        self.nn = NeuralNetwork([784,256,128,64,10])
        self.Z = np.array([[-1, 0], [1, -1], [0, 1]])

    def test_relU(self):
        expected = np.array([[0, 0], [1, 0], [0, 1]])
        np.testing.assert_array_equal(self.nn.relU(self.Z), expected)

    def test_drelU(self):
        expected = np.array([[0, 0], [1, 0], [0, 1]])
        np.testing.assert_array_equal(self.nn.drelU(self.Z), expected)

    def test_sigmoid(self):
        expected = np.array(
            [[0.26894142, 0.5], [0.73105858, 0.26894142], [0.5, 0.73105858]])
        np.testing.assert_allclose(self.nn.sigmoid(self.Z), expected, rtol=1e-05)

    def test_dsigmoid(self):
        expected = np.array(
            [[0.19661193, 0.25], [0.19661193, 0.19661193], [0.25, 0.19661193]])
        np.testing.assert_allclose(self.nn.dsigmoid(self.Z), expected, rtol=1e-05)

    def test_softmax(self):
        Z = np.array([1, 2, 3, 4, 1, 2, 3])
        expected = np.array([0.02364054, 0.06426166, 0.1746813,
                            0.474833, 0.02364054, 0.06426166, 0.1746813])
        np.testing.assert_allclose(self.nn.softmax(Z), expected, rtol=1e-6, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
