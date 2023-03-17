# TESTDOCUMENTATION

*This file contains low-level tests for various components of this neural network project. These tests are primarily meant to keep track of modifications and ensure the correct implementation of utility functions and data conversion.*

*It is important to note that these tests are not comprehensive and only provide a basic level of assurance. Some of the real tests, including tests for algorithms such as backpropagation are not implemented*

[![codecov](https://codecov.io/gh/JuusoSaavalainen/Neural-network-with-numpy/branch/main/graph/badge.svg?token=YO0Y9270ZS)](https://codecov.io/gh/JuusoSaavalainen/TiraLAB-Neural-network-with-numpy)

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

## TestFprop

This test file is for testing the forward propagation basic functionality in the neural network.

### Methods 

- setUp: sets up the test image as vector and the network with 2 hidden layers 10, 10
- test_forwardprop_wrong: test to ensure seeding
- test_forwardprop_right: test the outcome of the algorithm with fixed input and layout

### Attributes
- self.X_training: is an attribute that contains the training data, which is a numpy array with shape (784, 1). The data is reshaped to have 784 rows and 1 column using the np.reshape method.
- self.layers_dims: is a list that specifies the number of nodes in each layer of the neural network. In this case, the network has 3 layers with 784 nodes in the input layer, 10 nodes in the first hidden layer, and 10 nodes in the output layer.
- np.random.seed(42): sets the random seed for numpy's random number generator. This ensures that the same random values will be generated each time the program is run, making the results reproducible.
- nn: is an instance of the NeuralNetwork class with the specified layer dimensions. The weights of the network are randomly initialized using the seed.

## Testnnsup

This is the test file for basic supportive functions used in the model.
Contains 3 classes of unittests described here

### TestOneHot

- Test functionality of OneHot encoding with floats and large inputs

### TestNormalizeZeroOne

- Test the normalization from 0-255 to 0-1

### TestRandomizeRows

- Test the randomizing of the rows before training. 

## Missing tests
- Backwardpropagation and the training loop as whole
