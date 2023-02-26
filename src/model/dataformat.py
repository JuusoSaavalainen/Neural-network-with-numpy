import pandas as pd
import numpy as np


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


def format():
    # ::STILL IN DEVELOP MODE::

    # load the data as so path to train data is defined here that is why its here.
    # loading of the test data can be doned similarly

    # These
    data = pd.read_csv(
        'src/data/mnist_train.csv', header=None)

    # real testing data is here
    non_train_test = pd.read_csv(
        'src/data/mnist_test.csv', header=None)

    # randomize the pickd data
    data = randomize_rows(data)
    t_data = randomize_rows(non_train_test)

    # divide to validation set, now one column represents one picture.
    # validation set is not used for training thus its here if neeeded to evaluate results.
    validation_rate = 0.8
    training_size = round(data.shape[0] * validation_rate)
    training_data = data[:training_size:]
    trainval_data = data[training_size:, :]

    # define the labels and trainset , get RGB values to 0-1 interval.
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

    # testing data
    X_test = [x[1:] for x in t_data]  # vals
    X_test = np.array(X_test)
    Y_test = t_data[:, 0]  # labels

    # Reshape the array to 2-dimensional
    X_test = [np.reshape(i, (784, 1)) for i in X_test]
    X_test = np.array(X_test)

    # Change RGB Intreval to 0-1 from 0-255 for both sets
    X_training = normalize_zero_one(X_training)
    X_validating = normalize_zero_one(X_validating)
    X_test = normalize_zero_one(X_test)

    return X_training, Y_training, X_test, Y_test, X_validating, Y_validating
