# Recognition of handwritten numbers using machine learning methods

![GHA workflow badge](https://github.com/JuusoSaavalainen/TiraLAB-Neural-network-with-numpy/workflows/CI/badge.svg)
[![codecov](https://codecov.io/gh/JuusoSaavalainen/Neural-network-with-numpy/branch/main/graph/badge.svg?token=YO0Y9270ZS)](https://codecov.io/gh/JuusoSaavalainen/Neural-network-with-numpy)
## Model

Deep Neural network (DNN) implementation with MNIST db from scratch using NumPy. The layout of the network is set to be modular, so you can explore variations of layers and other hyperparameters with it. Generally i will be using 5 total layer setup. Regardless of the number of layers the model will be fully connected so we can call it MLP (Multilayer perceptron). Feel free to tune the model way you like. 

![](https://github.com/JuusoSaavalainen/Neural-network-with-numpy/blob/main/documentation/nnsimumnist.gif)

> *Gif demonstrating forward prop in the model with 2 hidden layers
[[source]](https://medium.com/analytics-vidhya/applying-ann-digit-and-fashion-mnist-13accfc44660)*

## GUI

If you dont wish to explore the training and the modularity of the model , you can test it with the GUI provided with the pretrained model.
There is simple Tkinter GUI implemented where you can draw your own digits. Those will be inputted to the model and you will be provided with the guess of the model. The implementation is not nearly as good as it could be, but with proper centered drawings it seems to preform really nice. The model included will be always the best current version possible. Applying some sort of centering algorithm to the picture would increase the accuracy regardless of the accuracy of the pretrained model *#TODO*.


#### Current model used in GUI: *test_data accuracy = 98.16%*

![](https://github.com/JuusoSaavalainen/Neural-network-with-numpy/blob/main/documentation/gui.gif)![](https://github.com/JuusoSaavalainen/Neural-network-with-numpy/blob/main/documentation/gui%20info.png)

## Training simple CLI

If you want you can train the model with your own parameters. Follow the instructions please. 

In the training you can choose custom values to these parameter:
- *Number of hidden layers* [positive number] ~Exmpl = 3
- *Size of those layers* [postive number] ~Exmpl = 256, 128 ,64
- *Learning rate* [positive number] ~Exmpl = 0.03
- *Epochs* [positive number] ~Exmpl = 10
- *Batch size* [positive number] ~Exmpl = 32
- *Activation function* [RelU, Sigmoid]

## Installing

You need to have Python installed with Poetry to be able to run this application. Clone the repository to your desired path.

```bash
# You should use
$ poetry shell

# Install dependencies
$ poetry install

# Run the GUI to test the pretrained model
$ poetry run invoke startgui

# Download the data to csv format
$ poetry run invoke dowloadmnist

# Run the taining cli
$ poetry run invoke train
```

## Docs
* [Specification document](https://github.com/JuusoSaavalainen/TiraLAB/blob/main/documentation/specification.md)
* [Test document](https://github.com/JuusoSaavalainen/TiraLAB-Neural-network-with-numpy/blob/main/documentation/testdocumentation.md)
* [Implementation document](https://github.com/JuusoSaavalainen/Neural-network-with-numpy/blob/main/documentation/implementation.md)
### Weekly raports
* [Week 1](https://github.com/JuusoSaavalainen/TiraLAB/blob/main/documentation/weeklyrecap1.md)
* [Week 2](https://github.com/JuusoSaavalainen/TiraLAB-Neural-network-with-numpy/blob/main/documentation/weeklyrecap2.md)
* [Week 3](https://github.com/JuusoSaavalainen/TiraLAB-Neural-network-with-numpy/blob/main/documentation/weeklyrecap3.md)
* [Week 4](https://github.com/JuusoSaavalainen/Neural-network-with-numpy/blob/main/documentation/weeklyrecap4.md)
* [Week 5](https://github.com/JuusoSaavalainen/Neural-network-with-numpy/blob/main/documentation/weeklyrecap5.md)
* [Week 6](https://github.com/JuusoSaavalainen/Neural-network-with-numpy/blob/main/documentation/weeklyrecap6.md)
