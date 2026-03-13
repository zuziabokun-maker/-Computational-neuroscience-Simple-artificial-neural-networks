# importing imports :p
import numpy as np
import scipy.special as ss


# Hyperparameters
learning_rate = 0.3
input_size = 3
hidden_size = 3
output_size = 3

#neural network class definition
class NeuralNetwork: 
    # initializing the neural network
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size  # the self statement, gives live to the variable as an object, even outside the function, so we can still access it even outside of its indentaded function
        self.hidden_size = hidden_size
        self.output_size = output_size

        # matrixes for input, hidden and output layers

        ## link between input and hidden layers 
        self.wih = np.random.rand(self.input_size, self.hidden_size) # the size of the matrix is the size of the input layer by the size of the hidden layer. this is this way becuase of matrix multiplication
        ## link between hidden and output layers
        self.who = np.random.rand(self.output_size, self.hidden_size)

        # activation function is the sigmoid function, read the signals from the hidden layer
        self.activation_function = lambda x: ss.expit(x)

        # learning rate
        self.lr = learning_rate
        pass
    
    # training the neural network
    def train(self):
        pass

    # querying the neural network
    def query(self, inputs):
        inputs = np.array(inputs, ndmin=2).T

        # To calculate the signals entering the hidden layer, we multiply the weight matrix
        # (between the input and hidden layers) by the input signals.
        hidden_inputs = np.dot(self.wih, inputs)
        
        # we calculate the signals emerging from the hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)

        # Calculate the signals emerging from final output
        final_outputs = self.activation_function(final_inputs)

        return final_outputs
        pass


# instance of neural network
n = NeuralNetwork(input_size, hidden_size, output_size)