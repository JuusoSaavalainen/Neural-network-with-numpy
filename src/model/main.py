import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#############
#  TiraLab  #
#  2023/III #
#  Author:  #
#  Juuso S  #
#           #
#############

#load the data as so
#csv -> df -> numpy.array -> shuffle -> transpose
data = pd.read_csv('/home/saavajuu/tiraLAB/data/mnist_train.csv', header=None) #pathing may change
data = np.array(data)
np.random.shuffle(data)
data = data.T

training_labs = data[0] #Labels ranging from 0 to 9
training_data = data[1:60000] #784 RGB values per col 
#print(training_labs)
#print(training_data[:, 0].shape)

class NN:
    def __init__(self):
        pass

    def forward_prop(self):
        pass

    def backward_prop(self):
        pass

    def activation_func(self):
        pass
        #sigmoid , relU , softmax , ..
    
