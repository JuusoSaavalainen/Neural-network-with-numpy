import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utility as utils


#############
#  TiraLab  #
#  2023/III #
#  Author:  #
#  Juuso S  #
#           #
#############


def main():
    """
    This is the main file to setup and train the Neural Network
    """
    # ::STILL IN DEVELOP MODE:: running this will train and test with 20 examples with pics and labels

    # load the data as so path to train data is defined here that is why its here.
    # loading of the test data can be doned similarly
    data = pd.read_csv(
        '/home/saavajuu/tiraLAB/src/data/mnist_train.csv', header=None)

    #real testing data is here
    non_train_test = pd.read_csv(
        '/home/saavajuu/tiraLAB/src/data/mnist_test.csv', header=None)


    # randomize the pickd data
    data = utils.randomize_rows(data)
    t_data = utils.randomize_rows(non_train_test)


    # divide to validation set and transpose it, now one column represents one picture.
    # validation set is not used for training thus its here if neeeded to evaluate results.
    validation_rate = 0.8
    training_size = round(data.shape[0] * validation_rate)
    training_data = data[:training_size:]
    trainval_data = data[training_size:, :]

    # define the labels and trainset , get RGB values to 0-1 interval.
    ##################################################################
    # training set
    X_training = [x[1:] for x in training_data]  # vals
    X_training = np.array(X_training)
    Y_training = training_data[:, 0]  # labels

    # Reshape the array to 2-dimensional
    X_training = [np.reshape(i, (784, 1)) for i in X_training]
    X_training = np.array(X_training)

    # validation set
    X_validating = [x[1:] for x in training_data]  # vals
    X_validating = np.array(X_training)
    Y_validating = training_data[:, 0]  # labels

    # Reshape the array to 2-dimensional
    X_validating = [np.reshape(i, (784, 1)) for i in X_validating]
    X_validating = np.array(X_validating)

    #testing data
    X_test = [x[1:] for x in t_data]  # vals
    X_test = np.array(X_test)
    Y_test = t_data[:, 0]  # labels

    # Reshape the array to 2-dimensional
    X_test = [np.reshape(i, (784, 1)) for i in X_test]
    X_test = np.array(X_test)

    # Change RGB Intreval to 0-1 from 0-255 for both sets
    X_training = utils.normalize_zero_one(X_training)
    X_validating = utils.normalize_zero_one(X_validating)
    X_test = utils.normalize_zero_one(X_test)

    # Set the wanted layers and other setable hyperparams
    #                                           ___________HIDDEN LAYERS I-III_____________
    #                                           | L1(in)  L2     L3       L4      L5(out) |
    # Right now im experiementing with this:     |  784  | 256 | 256/2 | 256/2/2 |    10   | ;model with 3 hidden layers.
    #                                           |_________________________________________|

    # these seems to get good outputs more tuning will be doned

    NN_layer_format = [784, 256, 128, 64, 10]
    learningrate = 0.1
    Epocs = 15

    # relU or sigmoid
    actication_func = ['relU', 'sigmoid']

    # if you want to see pics and labels during test forwarding
    visualize = False

    print(f'< Epoch goal: {Epocs} >')
    # Everything should be ready for training
    params = utils.gradient_descent(X_training, Y_training, NN_layer_format, Epocs,
                           learningrate, actication_func[0], X_test, Y_test)
    
    utils.test_model(X_test, Y_test, params, visualize)


if __name__ == '__main__':
    main()
