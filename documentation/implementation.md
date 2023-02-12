# Recognition of Handwritten Numbers

##  Goal
To be able to train your custom neural network , and test the model in practice

## Model

The model implementation is a Deep Neural Network (DNN) built from scratch using NumPy, with the MNIST database as the data source. The network architecture is designed to be modular, allowing for exploration of different layer configurations and hyperparameters. The current implementation that is used in GUI uses a 5-layer setup, with fully connected layers, making it a Multilayer Perceptron (MLP). 


## GUI

For those who do not wish to explore the training process or the modularity of the model, there is a Tkinter-based GUI provided for testing the pretrained model. The GUI allows the user to draw their own digits, which are then inputted to the model for prediction. The current implementation may not be optimal, but with properly centered drawings, the accuracy is quite high. Improving the centering algorithm for the input images could further increase the accuracy of the model. In GUI opencv libary is used for blurring the images. Pictures that are taken from the gui canvas should be close as possible to original mnist data for the best result. Blurring or smoothening the pixels is one step towards that. Centering of the pictures is something that should improve the model afterall (#TODO). 

## Training (CLI)
*only mvp avaible work in progress*

If desired, the model can be trained with custom parameters. The process involves cloning the repository and using Poetry to install dependencies and run the training CLI. The `idx_to_csv.py` file must be updated with the desired file paths before running `dowloadmnist` with Poetry. Once the CSV files are obtained, the paths must be updated in the `main.py` file. The training CLI can then be run with `poetry run invoke train` after correct paths are added to files. This is a bad solution but this week i didnt have time to think this and this works. 

In the training you can choose custom values to these parameter:
- *Number of hidden layers* [positive number]
- *Size of those layers* [postive number]
- *learning rate* [positive number]
- *Epochs* [positive number]
- *Batch size* [positive number] 
- *Activation function* [relU, sigmoid]

Training type could also be customize able. 

## Time and space complexity
todo

## Model Comparison 
todo
