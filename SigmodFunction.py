from numpy import *

def sigmoid(x):
    return 1 / (1 + exp(-x))

def sigmoid_derivative(x):
    return x * (1-x)

training_inputs = array([[0,0,1],
                         [1,1,1],
                         [1,0,1],
                         [0,1,1]])

training_outputs = array([[0,1,1,0]]).T

random.seed(1)

synaptic_weights = 2 * random.random((3, 1)) - 1

print('Random starting synaptic weights: ')
print(synaptic_weights)

for iteration in range(100000):
    input_layer = training_inputs

    outputs = sigmoid(dot(input_layer, synaptic_weights))

    error = training_outputs - outputs

    adjustments = error * sigmoid_derivative(outputs)

    synaptic_weights += dot(input_layer.T, adjustments) # adjusting weight

    print('Synaptic weights after training')
    print(synaptic_weights)



print('Output after training: ')
print(outputs)

