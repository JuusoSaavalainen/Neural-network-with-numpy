import os
import dataformat as dataformat
from utility import NeuralNetwork
import matplotlib.pyplot as plt

class NNCLI:
    def __init__(self):
        self.data = dataformat.format()
        self.X_training = self.data[0]
        self.Y_training = self.data[1]
        self.X_test = self.data[2]
        self.Y_test = self.data[3]
        self.X_validating = self.data[4]
        self.Y_validating = self.data[5]
        self.NN_layer_format = [784]
        self.batchsize = 0
        self.learningrate = 0
        self.Epocs = 0
        self.actication_func = None

    def clear(self):
        os.system('cls' if os.name == 'nt' else 'clear')

    def prompt_hidden_layers(self):
        while True:
            try:
                self.n_layers = int(input(
                    'How many hidden layers? [ Hidden layers are the layers in the network between in and output layers ] '))
                if self.n_layers <= 0:
                    raise ValueError(
                        "Number of hidden layers must be a positive integer.")
                break
            except ValueError:
                print("Please enter a positive integer for the number of hidden layers.")

        self.clear()

    def prompt_hidden_layer_sizes(self):
        for i in range(1, self.n_layers + 1):
            while True:
                try:
                    layer = int(
                        input(f"Size of the ({i}.) layer? [ Meaning the number of node in layer {i} ] "))
                    if layer <= 0:
                        raise ValueError(
                            "Size of each hidden layer must be a positive integer.")
                    self.NN_layer_format.append(layer)
                    break
                except ValueError:
                    print(
                        "Please enter a positive integer for the size of each hidden layer.")
            self.clear()
        self.NN_layer_format.append(10)

    def prompt_batch_size(self):
        while True:
            try:
                self.batchsize = int(input(
                    'Size of a batch? [ Training is divided to batches , bigger batch meaning faster training but less accuracy per step ] '))
                if self.batchsize <= 0:
                    raise ValueError("Batch size must be a positive integer.")
                break
            except ValueError:
                print("Please enter a positive integer for the batch size.")

        self.clear()

    def prompt_learning_rate(self):
        while True:
            try:
                self.learningrate = float(input(
                    'Learning rate / Alpha ? [ This is the size of a step (scalar) in gradient descent , Float example (0.03) ] '))
                if self.learningrate <= 0:
                    raise ValueError("Learning rate must be a positive float.")
                break
            except ValueError:
                print("Please enter a positive float for the learning rate.")

        self.clear()

    def prompt_epochs(self):
        while True:
            try:
                self.Epocs = int(input(
                    'Number of epochs? [ One epoch meaning one full training cycle with the whole training dataset ] '))
                if self.Epocs <= 0:
                    raise ValueError(
                        "Number of epochs must be a positive integer.")
                break
            except ValueError:
                print("Please enter a positive integer for the number of epochs.")

        self.clear()

    def prompt_activation_func(self):
        while True:
            self.actication_func = input(
                'Activation function? Choose [ReLU, Sigmoid]: [ This is math funct to add non-linearity to the nodes of the network ] ')
            if self.actication_func.lower() in ['relu', 'sigmoid']:
                break
            else:
                print(
                    "Please choose a valid activation function [ReLU, Sigmoid].")

        self.clear()

    def train(self):
        print(f'<< Your layout:  {self.NN_layer_format} <<')
        print(
            f'<< alpha = {self.learningrate}, batchsize = {self.batchsize}, epocs = {self.Epocs}, activation function = {self.actication_func} <<')
        print(f'<< Training starts <<')

        nn = NeuralNetwork(self.NN_layer_format)
        params = nn.gradient_descent_batch(self.X_training, self.Y_training, self.Epocs,
                                           self.learningrate, self.batchsize, self.actication_func)
        nn.test_model(self.X_test, self.Y_test, self.actication_func, False)

    def run(self):
        self.clear()
        self.prompt_hidden_layers()
        self.prompt_hidden_layer_sizes()
        self.prompt_batch_size()
        self.prompt_learning_rate()
        self.prompt_epochs()
        self.prompt_activation_func()
        self.train()