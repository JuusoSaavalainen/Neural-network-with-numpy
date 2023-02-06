# Numpy is used to ease the calculation process of all algos in this file
import numpy as np

# seaborn and matplotlib are purely for plotting to visualize results
import matplotlib.pyplot as plt
import seaborn as sns
import time 
#######################################################################
# _______________DATA HANDLERS / FORMATTER / ENCODERS__________________#


def init__nn(layers_dims):
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
    for i in range(1, len(layers_dims)):
        params[f'W{i}'] = np.random.randn(
            layers_dims[i], layers_dims[i-1]) * np.sqrt(1/layers_dims[i])
        params[f'b{i}'] = np.random.randn(
            layers_dims[i], 1) * np.sqrt(1/layers_dims[i])
    return params


def one_hot(Y):
    one_hot_Y = np.zeros(10)
    one_hot_Y[Y] = 1
    return one_hot_Y.reshape(10, 1)


def normalize_zero_one(data):
    """
    Function to push RGB values from 0-255 to interval 0-1

    Args:
        data np.array: dataset used 

    Returns:
        normalized np.array: np.array/255
    """
    norms = data/255.
    return norms


def randomize_rows(data):
    """
    This is the first function that gets called to and its 
    used for randomize the given dataset
    """
    df = np.array(data)
    np.random.shuffle(df)
    return df


#######################################################################
# _____________________ACTIVATION_FUNCS________________________________#
# ______________more can be added to test diff between_________________#

def relU(Z):
    """
    relU : Computes the RelU activation function
    Args:
        Z (numpy array or scalar): values to apply the func

    Returns:
        numpy array / scalar applied the funct
    """
    return np.maximum(Z, 0)


def drelU(Z):
    """
    drelU : Computes the derivate of RelU activation function
    Args:
        Z (numpy array or scalar): values to apply the func

    Returns:
        numpy array / scalar applied the funct
    """
    return Z > 0


def sigmoid(z):
    """
    Inputs:
    z: numpy array, shape (n, m)

    Output:
    a: numpy array, shape (n, m), containing sigmoid of the input

    """
    a = 1 / (1 + np.exp(-z))
    return a


def dsigmoid(z):
    """
    Inputs:
    z: numpy array, shape (n, m)

    Output:
    da: numpy array, shape (n, m), containing derivative of sigmoid of the input

    """
    s = sigmoid(z)
    da = s * (1 - s)
    return da


def softmax(Z):
    """
    Softmax : Computes the softmax activation function
    Args:
        Z (numpy array or scalar): values to apply the func

    Returns:
        numpy array / scalar applied the softmax funct
    """
    expZ = np.exp(Z) / sum(np.exp(Z))
    return expZ


#######################################################################
# _____________________BACK_&_FORWARD PROPAGATION_______________________#

def forwardprop(X, parameters, activation_function='relU'):
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
    n_layers = len(parameters) // 2

    activations = {'A0': X}

    for i in range(1, n_layers):
        activations[f'Z{i}'] = np.dot(
            parameters[f'W{i}'], activations[f'A{i-1}']) + parameters[f'b{i}']
        if activation_function == 'relU':
            activations[f'A{i}'] = relU(activations[f'Z{i}'])
        elif activation_function == 'sigmoid':
            activations[f'A{i}'] = sigmoid(activations[f'Z{i}'])
        else:
            raise ValueError(
                f"Invalid activation function: {activation_function}. Supported functions are 'relU' and 'sigmoid'.")

    activations['Z' + str(n_layers)] = np.dot(parameters['W' + str(n_layers)],
                                              activations['A' + str(n_layers - 1)]) + parameters['b' + str(n_layers)]
    activations['A' + str(n_layers)
                ] = softmax(activations['Z' + str(n_layers)])

    return activations


def backprop(activations, parameters, labels, activation_function='relU'):
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
    one_hot_labels = one_hot(labels)
    # hardcoded thinks that training will be with whole dataset this will not stay
    trainingsize = 48000
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
                parameters[f'W{layer+1}'].T, derivatives[f'dZ{layer+1}']) * drelU(activations[f'Z{layer}'])
        elif activation_function == 'sigmoid':
            derivatives[f'dZ{layer}'] = np.dot(
                parameters[f'W{layer+1}'].T, derivatives[f'dZ{layer+1}']) * dsigmoid(activations[f'Z{layer}'])
        else:
            raise ValueError(
                f"Invalid activation function: {activation_function}. Supported functions are 'relU' and 'sigmoid'.")

        gradients[f'dW{layer}'] = 1 / trainingsize * \
            np.dot(derivatives[f'dZ{layer}'], activations[f'A{layer-1}'].T)
        gradients[f'db{layer}'] = 1 / trainingsize * \
            np.sum(derivatives[f'dZ{layer}'], axis=1, keepdims=True)

    return gradients

#######################################################################
# ___________GRADIEN_DESC, ACCURACY, PREDICTS, MONITORING______________#

# todo


def update_parameters(params, grads, alpha):
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
    params_updated = {key: params[key] - alpha *
                      grads[f'd{key}'] for key in params.keys()}
    return params_updated


def compute_accuracy(predictions, targets):
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


def compute_loss(predictions, targets):
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


def gradient_descent(X, Y, layers_dims, max_iter, alpha, actifunc, X_test, Y_test):
    """
    Optimizes the neural network parameters using gradient descent optimization algorithm.
    This is the training loop of the nn. 

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

    # initialize parameters
    params = init__nn(layers_dims)
    L = len(params) // 2
    accuracies, losses = [], []

    # iterate through the optimization
    for iteration in range(1, max_iter + 1):
        for x, y in zip(X, Y):

            # forward propagation to compute activations
            activations = forwardprop(x, params, actifunc)

            # make predictions
            predictions = activations[f'A{L}']

            # backpropagation to compute gradients
            gradients = backprop(activations, params, y, actifunc)

            # update parameters using gradients
            params = update_parameters(params, gradients, alpha)

            # append accuracy and loss to their respective lists
            accuracy = compute_accuracy(predictions, one_hot(y))
            loss = compute_loss(predictions, one_hot(y))
            accuracies.append(accuracy)
            losses.append(loss)

        # print accuracy and loss after each epoch
        print("Epoch:", iteration, "Accuracy:", np.mean(
            accuracies), "Loss:", np.mean(losses))
        # reset the lists for accuracy and loss after each epoch
        accuracies, losses = [], []
        progress = (iteration + 1) / max_iter +1
        bar_length = int(progress * 50)
        bar = "=" * bar_length
        percent = int(progress * 100)
        print(f"[{bar:<50}] {percent}%", end='\r')
        time.sleep(0.1)
    print("[==================================================] 100%")
    return params

def test_model(x,y,params,visualize):
    # after training test with test data
    i = 0
    test_acc = 0
    rounds = 0
    for x, y in zip(x, y):
        activations = forwardprop(x, params)
        predictions = activations[f'A4']

        if visualize == True:
            # Plot the input image
            plt.imshow(x.reshape(28, 28), cmap='gray')
            plt.title(f'Prediction: {np.argmax(predictions)}, Real label: {y}')

            plt.show()
            # Display the whole vector of predictions
            print(f'Prediction vector: {predictions}')

        if np.argmax(predictions) == y:
            test_acc += 1
        rounds +=1
    print(test_acc/rounds)

    return params
