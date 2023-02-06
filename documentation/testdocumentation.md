# TESTDOCUMENTATION

*This file contains low-level tests for various components of this neural network project. These tests are primarily meant to keep track of modifications and ensure the correct implementation of utility functions and data conversion.*

*It is important to note that these tests are not comprehensive and only provide a basic level of assurance. The real tests, including tests for the neural network itself and algorithms such as backpropagation and feedforward, will be added in the future. These tests will be the main focus and will provide a more complete assessment of the project's functionality.*

[![codecov](https://codecov.io/gh/JuusoSaavalainen/TiraLAB-Neural-network-with-numpy/branch/main/graph/badge.svg?token=YO0Y9270ZS)](https://codecov.io/gh/JuusoSaavalainen/TiraLAB-Neural-network-with-numpy)

## TestConvert

This test file is for testing the conversion of IDX files to CSV files.

### Methods

- setUp: sets up the test files and creates mock image and label IDX files.
- test_convert: tests the conversion of IDX files to CSV files.
- tearDown: removes the mock IDX and CSV files after the test is finished.

### Attributes

- image: a string representing the filename of the mock image IDX file.
- label: a string representing the filename of the mock label IDX file.
- out: a string representing the filename of the output CSV file.
- n: an integer representing the number of samples in the mock IDX files.

## TestActivationFunctions

This test file is for testing the activation functions of the neural network.

### Methods

- setUp: sets up the Z matrix used in the tests.
- test_relU: tests the Rectified Linear Unit (ReLU) activation function.
- test_drelU: tests the derivative of the ReLU activation function.
- test_sigmoid: tests the sigmoid activation function.
- test_dsigmoid: tests the derivative of the sigmoid activation function.
- test_softmax: tests the softmax activation function.

### Attributes

- Z: a matrix used as input in the tests of the activation functions.

