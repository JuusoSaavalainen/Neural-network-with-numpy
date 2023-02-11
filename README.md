# Recognition of handwritten numbers using machine learning methdos

![GHA workflow badge](https://github.com/JuusoSaavalainen/TiraLAB-Neural-network-with-numpy/workflows/CI/badge.svg)
[![codecov](https://codecov.io/gh/JuusoSaavalainen/TiraLAB-Neural-network-with-numpy/branch/main/graph/badge.svg?token=YO0Y9270ZS)](https://codecov.io/gh/JuusoSaavalainen/TiraLAB-Neural-network-with-numpy)
## Model

Deep Neural network (DNN) implementation with MNIST db from scratch using NumPy. The layout of the network is set to be modular, so you can explore variations of layers and other hyperparameters with it. Generally i will be using 5 total layer setup. Regardless of the number of layers the model will be fully connected so we can call it MLP (Multilayer perceptron). Feel free to tune the model way you like. 

![](https://github.com/JuusoSaavalainen/Neural-network-with-numpy/blob/main/documentation/nnsimumnist.gif)

> *Gif demonstrating forward prop in the model with 2 hidden layers
[[source]](https://medium.com/analytics-vidhya/applying-ann-digit-and-fashion-mnist-13accfc44660)*

## GUI

If you dont wish to explore the training and the modularity of the model , you can test it with the GUI provided with the pretrained model.
There is simple Tkinter GUI implemented where you can draw your own digits. Those will be inputted to the model and you will be provided with the guess of the model. The implementation is not nearly as good as it could be, but with proper centered drawings it seems to preform really nice. The model included will be always the best current version possible. Applying some sort of centering algorithm to the picture would increase the accuracy regardless of the accuracy of the pretrained model *#TODO*.

### Current model used in GUI: *test_data accuracy = 0.9517*

![](https://github.com/JuusoSaavalainen/Neural-network-with-numpy/blob/main/documentation/gui.gif)  |  ![](https://github.com/JuusoSaavalainen/Neural-network-with-numpy/blob/main/documentation/gui%20info.png)

## Installing

You need to have Python installed with Poetry to be able to run this application. Clone the repository to your desired path.

```bash
# Install dependencies
$ poetry install

# Run the GUI to test the pretrained model
$ poetry run invoke startgui

# Run the taining with your own model structure
# First set the wanted hyperparameters to main.py
$ poetry run invoke train
```

## Docs
* [Specification document](https://github.com/JuusoSaavalainen/TiraLAB/blob/main/documentation/specification.md)
* [Test document](https://github.com/JuusoSaavalainen/TiraLAB-Neural-network-with-numpy/blob/main/documentation/testdocumentation.md)

### Weekly reaports
* [Week 1](https://github.com/JuusoSaavalainen/TiraLAB/blob/main/documentation/weeklyrecap1.md)
* [Week 2](https://github.com/JuusoSaavalainen/TiraLAB-Neural-network-with-numpy/blob/main/documentation/weeklyrecap2.md)
* [Week 3](https://github.com/JuusoSaavalainen/TiraLAB-Neural-network-with-numpy/blob/main/documentation/weeklyrecap3.md)
