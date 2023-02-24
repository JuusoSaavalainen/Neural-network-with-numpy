from cli import NNCLI

def main():
    cli = NNCLI()
    cli.run()


if __name__ == '__main__':
    main()

#
#    NN_layer_format = [784]
#    n_layers = int(input('How many hidden layers? '))
#    for i in range(1, n_layers + 1):
#        layer = int(input(f'Size of the ({i}.) layer? '))
#        NN_layer_format.append(layer)
#    NN_layer_format.append(10)
#
#    batchsize = int(input('Size of a batch? '))
#
#    # learning rate sometimes referred as alpha is the desired scalar to move in the gradient descent optimizing
#    learningrate = float(input('Learning rate / alpha ? '))
#
#    # one epoch means training once with the whole training data
#    Epocs = int(input('Number of epocs ? '))