import numpy as np

np.random.seed(0)

# forward objects serve to calculate the outputs obtained in each step for the next usecase
# bacward objects will calculate the outputs that will change the own neurons' changes 

# Neuron object:
class Layer_Dense:

    def __init__(self, n_inputs, n_neurons):
        # the layer will be started with the number of neurons provided
        # and the number of inputs that can either be Raw data or results
        # from another layer
        # wich can be adjusted

        # setting random weights with a seed
        # starts the weights in a way that we dont need to transpose
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons) 

        # starts the biases in a zeroe'd matrix, so we can update it later
        self.biases = np.zeros((1, n_neurons))
        

    def forward(self, inputs):
        # Multiply the inputs by the weights and adds bias
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        
        # will return a matrix in the form of [ n_neurons x n_inputs]
        # containing the neurons' result for each information given 
        # this being [(characteristic * weight) + bias]  
        return(self.output)
    
    def backward(self, dvalues):
        # 
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)
    

# Loss Objects
class loss:
    def calculate(self, output , y_expected):
        # output is the outputs from the neurons
        # y is the expected result
        sample_losses = self.forward(output, y_expected)
        data_loss = (np.mean(sample_losses))
        return(data_loss)


class loss_Categorical_Cross_Entropy(loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        # clipping the loss not to reach infinity a 0% compatible result
        #  in a single prediction would lead to infinite loss across the board
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
        # this will return a vector for each result
            correct_confidences = y_pred_clipped[range(samples), y_true]

        # This part is unescessary for this project, but since I intend to learn the 
        # correct way to deal with data, this will be kept for a possible future reference
        # this will deal with the y_true making the internal product of each row
        elif len(y_true.shape) ==2:
        # this will return a vector for each result
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)

        # loss calculated via the *negative log* of the certainty of each prediction 
        negative_log_likelyhoods = -np.log(correct_confidences)
        return (negative_log_likelyhoods)
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(y_true.shape)==1:
            y_true = np.eye(labels)[y_true]
        
        self.dinputs = -y_true/dvalues
        
        self.dinputs = self.dinputs/samples

# Activation functions objects:
class activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0,inputs)
        return self.output
    
    def backward(self, dvalues):
        # Since we need to modify the original variable,
        # let's make a copy of the values first
        self.dinputs = dvalues.copy()
        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0
    
class activation_Step:
    def forward(self, inputs):
        # self.output = np.maximum(0,inputs)
        self.output = np.where(inputs > 0, 1, 0)
        return self.output

class activation_softmax:
    def forward(self, inputs):
        # exp values will exponentiate each value on every batch
        # subtracting the max value of each in it's respective batch
        # keeping exponents between 0 and 1 and keeping vector dimensions 
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) 

        # softmax function keeping the vector dimensions
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        return self.output

    def backward(self, dvalues):
        # starts unitialized array
        self.dinputs = np.empty_like(dvalues)
        # enumerates outputs and gradients
        for i, (single_output, single_dvalues) in enumerate (zip(self.output, dvalues)):
            # Reshape to flatten the output array 
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output and calculate samples gradient
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output,single_output.T)
            # Adding it to the array of sample gradients
            self.dinputs = np.dot(jacobian_matrix, single_dvalues)

# Softmax classifier - combined Softmax activation
# and cross-entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy():
    # Creates activation and loss function objects
    def __init__(self):
        self.activation = activation_softmax()
        self.loss = loss_Categorical_Cross_Entropy()
        # Forward pass

    def forward(self, inputs, y_true):
        # Output layer's activation function
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output
        # Calculate and return loss value
        return self.loss.calculate(self.output, y_true)
    
    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of dvaluesamples

        samples = len(dvalues)

        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples


# the book suggests to use a learning rate decay, but I chose not to
# since the problem we are solving is quite simple
class Optimizer_SGD:
# Initialize optimizer - set settings,
# learning rate of 1. is default for this optimizer
    def __init__(self, learning_rate=1.0):
     self.learning_rate = learning_rate
     self.iterations = 0

     def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
            (1. / (1. + self.decay * self.iterations))
        
    # Update parameters
    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases
        self.iterations += 1

    # Call after any parameter updates
    def post_update_params(self):
        pass

    