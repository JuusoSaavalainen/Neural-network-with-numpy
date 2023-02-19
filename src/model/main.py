from utility import NeuralNetwork
import gui as gui
import dataformat as dataformat

def main():
    data = dataformat.format()
    X_training = data[0]
    Y_training = data[1]
    X_test = data[2]
    Y_test = data[3]
    X_validating = data[4]
    Y_validating = data[5]


    NN_layer_format = [784]
    n_layers = int(input('How many hidden layers? '))
    for i in range(1, n_layers + 1):
        layer = int(input(f'Size of the ({i}.) layer? '))
        NN_layer_format.append(layer)
    NN_layer_format.append(10)

    batchsize = int(input('Size of a batch? '))

    # learning rate sometimes referred as alpha is the desired scalar to move in the gradient descent optimizing
    learningrate = float(input('Learning rate / alpha ? '))

    # one epoch means training once with the whole training data
    Epocs = int(input('Number of epocs ? '))
    actication_func = (input('Actuvation function ? choose [relU , sigmoid]: '))


    # Everything should be ready for training

    print(f'<< Your layout:  {NN_layer_format} <<')
    print(f'<< alpha = {learningrate}, batchsize = {batchsize}, epocs = {Epocs}, activation funtion = {actication_func} <<')
    print(f'<< Training starts <<')
    
    nn = NeuralNetwork(NN_layer_format)
    params = nn.gradient_descent_batch(X_training, Y_training, Epocs,
                           learningrate, batchsize, actication_func)

    nn.test_model(X_test, Y_test, actication_func, False)


if __name__ == '__main__':
    main()
