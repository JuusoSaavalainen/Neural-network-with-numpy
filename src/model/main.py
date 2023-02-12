import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utility as utils
import gui as gui
import pickle

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
    # ::STILL IN DEVELOP MODE:: 

    # load the data as so path to train data is defined here that is why its here.
    # loading of the test data can be doned similarly

    #These
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

    #########################################################################################################################################################
    #                                                                                                                                                       #
    # Set the wanted layers and other setable hyperparams                                                                                                   #
    #                  333                         ___________HIDDEN LAYERS I-III_____________                                                                 #
    #                                           | L1(in)  L2     L3       L4      L5(out) |                                                                 #
    # Right now im experiementing with this:    |  784  | 256 | 256/2 | 256/2/2 |    10   | ;model with 3 hidden layers.                                    #
    #                                           |_________________________________________|                                                                 #
    #                                                                                                                                                       #
    # these seems to get good outputs more tuning will be doned                                                                                             #
    #                                                                                                                                                       #
    # with alpha = 0.5 , it seems like going above 25 epoch does not change much and above 90 can be found around 10 epoch , 25 epoch yielded 0.02 better.  #
    # alpha 0.5 :: 15epoch = 0.9191 , 20epoch = 0.9223, 25epoch = 0.926 , 50epoch = 0.9384, 75epoch = 0.9476// 200 = 0.9517 #W TEST DATA                    #
    # it takes about 5080 sek to train 75 epoch, with batching = sized 190 epoch 200 === testacc == 0.9601. with batching size 10 and epochs =150 tacc=0.9774                                                                                                              #
    #                                                                                                                                                       #
    #           0.9801 with layer above lr = 0.02 epoch 150 batching in 10 samples                                                                                               #
    #########################################################################################################################################################

    # __HERE_YOU_CAN_SET_THE_FORMAT_OF_LAYERS_AND_OTHER_HYPERPARAMS!!!!__
    # setable: nn format , learningrate , epocs , actifunct, visualize

    # this is format of the network firs and last layer needs to be 784 and 10, but the amount of nodes
    # and the amount of those hidden layers between in and output can be anything, set it how you like
    NN_layer_format = [784]
    n_layers = int(input('How many hidden layers? '))
    for i in range(1, n_layers + 1):
        layer = int(input(f'Size of the ({i}.) layer? '))
        NN_layer_format.append(layer)
    NN_layer_format.append(10)

    batchsize = int(input('Size of a batch? '))

    # learning rate sometimes referred as alpha is the desired scalar to move in the gradient descent optimizing

    learningrate = float(input('Learning rate / alpha ? '))

    # one epcoch means training once with the whole training data

    Epocs = int(input('Number of epocs ? '))
    #choose relU or sigmoid

    actication_func = (input('Actuvation function ? choose [relU , sigmoid]: '))

    # if you want to see pics and labels during test forwarding set this True
    #visualize = (input('Visualize testing data after training ? type [ y ] to activate: '))

    # Everything should be ready for training

    print(f'<< Your layot:  {NN_layer_format} <<')
    print(f'<< alpha = {learningrate}, batchsize = {batchsize}, epocs = {Epocs}, activation funtion = {actication_func} <<')
    print(f'<< Training starts <<')
    
    params = utils.gradient_descent_batch(X_training, Y_training, NN_layer_format, Epocs,
                           learningrate, batchsize, actication_func)


    # This will run the test data trought the model and calculate error with unseen data agter training
    #if visualize == 'y':
    #    utils.test_model(X_test, Y_test, params, True)

    #if visualize != "y":
    utils.test_model(X_test, Y_test, params, False)
    # this will be used to capture the calculated weights if we want to save the model



    #with open('NEWQ_trained150e.pickle', 'wb') as f:
    #    pickle.dump(params, f)

if __name__ == '__main__':
    main()
