import numpy as np
import matplotlib as plt
import pickle

class NeuralNetwork:

    def __init__(self, layer_dims):
        self.layer_dims = layer_dims
        self.params = self.init__nn()

    def load_params(self):
        with open('src/model/model98.pickle', 'rb') as f:
            self.params = pickle.load(f)
        return self.params
        
    def init__nn(self):
        """
        Initalizes parameters for the NN
        to be precise it randomizez and initializez all
        the weights and biases for the beginning of the training

        Dicts are used to store these values since we dont
        want to hardcode this part to keep it layer layout modular.

        W denotes weight matrix between nodes
        b denotes bias vecktor
        Returs dict with W1, b1, W2, b2 ..... Wn, bn 
        where n denotes the dimension of given model
        """
        params = {}
        for i in range(1, len(self.layer_dims)):
            params[f'W{i}'] = np.random.randn(
                self.layer_dims[i], self.layer_dims[i-1]) * np.sqrt(1/self.layer_dims[i])
            params[f'b{i}'] = np.random.randn(
                self.layer_dims[i], 1) * np.sqrt(1/self.layer_dims[i])
        return params


    def one_hot(self, Y):
        one_hot_Y = np.zeros(10)
        one_hot_Y[Y] = 1
        return one_hot_Y.reshape(10, 1)

    def relU(self, Z):
        """
        relU : Computes the RelU activation function
        Args:
            Z (numpy array or scalar): values to apply the func

        Returns:
            numpy array / scalar applied the funct
        """
        return np.maximum(Z, 0)


    def drelU(self, Z):
        """
        drelU : Computes the derivate of RelU activation function
        Args:
            Z (numpy array or scalar): values to apply the func

        Returns:
            numpy array / scalar applied the funct
        """
        return Z > 0


    def sigmoid(self, z):
        """
        Inputs:
        z: numpy array, shape (n, m)

        Output:
        a: numpy array, shape (n, m), containing sigmoid of the input

        """
        a = 1 / (1 + np.exp(-z))
        return a


    def dsigmoid(self, z):
        """
        Inputs:
        z: numpy array, shape (n, m)

        Output:
        da: numpy array, shape (n, m), containing derivative of sigmoid of the input

        """
        s = self.sigmoid(z)
        da = s * (1 - s)
        return da


    def softmax(self, Z):
        """
        Softmax : Computes the softmax activation function
        Args:
            Z (numpy array or scalar): values to apply the func

        Returns:
            numpy array / scalar applied the softmax funct
        """
        #stabilizer
        Z = Z - max(Z)

        expZ = np.exp(Z) / sum(np.exp(Z))
        return expZ

    def forwardprop(self, X, activation_function='relU'):
        """
        Implements the forward propagation algorithm for a neural network.

        Parameters:
        X (numpy.ndarray): Input data of shape (number of features, number of examples).
        parameters (dict): Dictionary containing the parameters (weights and biases) for each layer of the network. 
            The keys should be in the format 'W1', 'b1', 'W2', 'b2', ..., 'WL', 'bL', where L is the number of layers.
        activation_function (str, optional): The activation function to be used for the hidden layers. Can be 'relU' or 'sigmoid'. Default is 'relU'.

        Returns:
        activations (dict): Dictionary containing the activations for each layer of the network.
            The keys should be in the format 'A0', 'Z1', 'A1', 'Z2', 'A2', ..., 'AL-1', 'ZL', 'AL', where L is the number of layers.
        """
        n_layers = len(self.params) // 2

        activations = {'A0': X}

        for i in range(1, n_layers):
            activations[f'Z{i}'] = np.dot(
                self.params[f'W{i}'], activations[f'A{i-1}']) + self.params[f'b{i}']
            if activation_function == 'relU':
                activations[f'A{i}'] = self.relU(activations[f'Z{i}'])
            elif activation_function == 'sigmoid':
                activations[f'A{i}'] = self.sigmoid(activations[f'Z{i}'])
            else:
                raise ValueError(
                    f"Invalid activation function: {activation_function}. Supported functions are 'relU' and 'sigmoid'.")

        activations['Z' + str(n_layers)] = np.dot(self.params['W' + str(n_layers)],
                                                activations['A' + str(n_layers - 1)]) + self.params['b' + str(n_layers)]
        activations['A' + str(n_layers)
                    ] = self.softmax(activations['Z' + str(n_layers)])

        return activations


    def backprop(self, activations, parameters, labels, sizeb, activation_function='relU'):
        """
        the main algo used in optimizing the NN


        Parameters:
        activations - a dictionary in the format {'A0':..., 'A1':..., 'Z1':..., 'A2':..., ...}
        parameters - a dictionary in the format {'W1':..., 'b1':..., 'W2':...}
        labels - the target values (Y)
        activation_function - the activation function used in the forward propagation, e.g. 'relu' or 'sigmoid'
        Returns:
        gradients - a dictionary in the format {'dW1':..., 'db1':..., ...}
        """

        num_layers = len(parameters) // 2
        one_hot_labels = self.one_hot(labels)
        # hardcoded thinks that training will be with whole dataset this will not stay
        trainingsize = sizeb
        derivatives, gradients = {}, {}

        # For the last layer (no actvation func derivate needed)
        derivatives['dZ' + str(num_layers)] = activations['A' +
                                                        str(num_layers)] - one_hot_labels
        gradients['dW' + str(num_layers)] = 1 / trainingsize * np.dot(
            derivatives['dZ' + str(num_layers)], activations['A' + str(num_layers - 1)].T)
        gradients['db' + str(num_layers)
                ] = np.sum(derivatives['dZ' + str(num_layers)])

        # For layers (L-1) to 1 (action func derivate needed)
        for layer in reversed(range(1, num_layers)):
            if activation_function == 'relU':
                derivatives[f'dZ{layer}'] = np.dot(
                    parameters[f'W{layer+1}'].T, derivatives[f'dZ{layer+1}']) * self.drelU(activations[f'Z{layer}'])
            elif activation_function == 'sigmoid':
                derivatives[f'dZ{layer}'] = np.dot(
                    parameters[f'W{layer+1}'].T, derivatives[f'dZ{layer+1}']) * self.dsigmoid(activations[f'Z{layer}'])
            else:
                raise ValueError(
                    f"Invalid activation function: {activation_function}. Supported functions are 'relU' and 'sigmoid'.")

            gradients[f'dW{layer}'] = 1 / trainingsize * \
                np.dot(derivatives[f'dZ{layer}'], activations[f'A{layer-1}'].T)
            gradients[f'db{layer}'] = 1 / trainingsize * \
                np.sum(derivatives[f'dZ{layer}'], axis=1, keepdims=True)

        return gradients

    def update_parameters(self, params, grads, alpha):
        """
        Update the parameters (weights and biases) of a neural network.
        implemented using dict comprehension since dicts are used mostly in this project.
        Arguments:
        params -- a dictionary containing the current parameters (weights and biases) of a neural network
        grads -- a dictionary containing the gradient of the parameters (weights and biases) of a neural network
        alpha -- the learning rate, a scalar value determining the step size for updating the parameters

        Returns:
        params_updated -- a dictionary containing the updated parameters (weights and biases) of the neural network
        """
        self.params = {key: self.params[key] - alpha *
                        grads[f'd{key}'] for key in self.params.keys()}
        return self.params


    def compute_accuracy(self, predictions, targets):
        """
        Computes the accuracy between the predictions and targets.

        Parameters:
        predictions - predicted values from the model
        targets - actual target values

        Returns:
        accuracy - the computed accuracy
        """
        accuracy = np.mean(np.round(predictions) == targets)
        return accuracy


    def compute_loss(self, predictions, targets):
        """
        Computes the mean squared error between the predictions and targets.

        Parameters:
        predictions - predicted values from the model
        targets - actual target values

        Returns:
        loss - the computed loss
        """
        loss = np.mean((predictions - targets) ** 2)
        return loss


    def gradient_descent_batch(self, X, Y , max_iter, alpha, batchsize, actifunc):
        """
        Optimizes the neural network parameters using gradient descent optimization algorithm.
        This is the training loop of the nn , this is not the smartest way since it optimizez after every sample.
        this could be changet later to stocastic or batch type implemention of training. 

        Parameters:
        X - input data
        Y - target values
        layers_dims - list of layer dimensions, including input and output layer
        max_iter - maximum number of iterations for optimization
        alpha - learning rate
        actifuc - chosen activation function name 

        x_train, y_train - temporary params here

        Returns:
        params - optimized parameters for the neural network
        """

        L = len(self.params) // 2
        accuracies, losses = [], []


        # iterate through the optimization in batches
        for iteration in range(1, max_iter + 1):
            
            data = list(zip(X,Y))
            np.random.shuffle(data)

            #baching
            mini_batches = [data[j:j + batchsize] for j in range(0, 48000, batchsize)]

            for batch in mini_batches:
                x,y = batch[0][0],batch[0][1]

                # forward propagation to compute activations
                activations = self.forwardprop(x, actifunc)

                # make predictions
                predictions = activations[f'A{L}']

                # backpropagation to compute gradients
                gradients = self.backprop(activations, self.params, y, batchsize,  actifunc)

                # update parameters using gradients 
                self.params = self.update_parameters(self.params, gradients, alpha)

                # append accuracy and loss to their respective lists
                accuracy = self.compute_accuracy(predictions, self.one_hot(y))
                loss = self.compute_loss(predictions, self.one_hot(y))
                accuracies.append(accuracy)
                losses.append(loss)

            # print accuracy and loss after each epoch
            print(f'Epoch -> {iteration:3} / {max_iter:1} | Training accuracy:{np.mean(accuracies):20} | Loss: {np.mean(losses):14}')
            # reset the lists for accuracy and loss after each epoch
            accuracies, losses = [], []
        return self.params


    def test_model(self, x, y, actf, visualize):
        # after training test with test data
        i = 0
        test_acc = 0
        rounds = 0
        correct_pics = []
        num_layers = len(self.params) // 2

        for x, y in zip(x, y):
            activations = self.forwardprop(x, actf)
            predictions = activations[f'A{num_layers}'] #stupid hardcore debugging

            if visualize == True and i < 50:
                # Plot the input image
                plt.imshow(x.reshape(28, 28), cmap='gray')
                plt.title(f'Prediction: {np.argmax(predictions)}, Real label: {y}')

                plt.show()
                # Display the whole vector of predictions
                print(f'Prediction vector: {predictions}')

            if np.argmax(predictions) == y:
                test_acc += 1
                correct_pics.append(x)
            rounds +=1
            i +=1
        if (test_acc/rounds)*100 <= 10:
            print(f'Your model got {(test_acc/rounds)*100}% right with the Test_data not used in training.')
            print('I can guess better than that model')

        print(f'Your model got {(test_acc/rounds)*100}% right with the Test_data not used in training.')