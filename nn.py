import numpy as np
from typing import Callable,List

# single layer of a neural network
class Layer:
        def __init__(self, input_size:int, num_neurons:int, activation_function:Callable[[np.ndarray],np.ndarray],activation_derivative: Callable[[np.ndarray], np.ndarray]):
                self.weights = np.random.rand(input_size,num_neurons)
                self.biases = np.zeros((1,num_neurons))
                self.activation_function = activation_function
                self.activation_derivative = activation_derivative

        def forward_pass(self, inputs:np.ndarray)->np.ndarray:
                self.inputs = inputs
                self.forward = np.dot(inputs, self.weights) + self.biases
                self.activated = self.activation_function(self.forward)
                return self.activated
        
        def backpropagate(self, da:np.ndarray, learning_rate:float)->np.ndarray:
                numrows = da.shape[0]

                dz = self.activation_derivative(self.forward)*da
                dw = np.matmul(self.inputs.T,dz)/numrows
                db = np.matmul(np.ones((numrows,1)).T,dz)/numrows

                self.weights = self.weights - learning_rate*dw
                self.biases = self.biases - learning_rate*db

                next_da = np.dot(dz, self.weights.T)
                return next_da

# multi-layer ann
class NeuralNetwork:
        def __init__(self, layers:List[Layer],loss_derivative:Callable, learning_rate:float=1.0):
                self.layers = layers
                self.learning_rate = learning_rate
                self.loss_derivative = loss_derivative
        
        def train(self, iterations:int ,training_data:np.ndarray):
                labels = training_data[:,[0]]
                for i in range(iterations):
                        features = training_data[:, 1:]
                        for layer in self.layers:
                                features = layer.forward_pass(features)
                        #features become the outputs 
                        da = self.loss_derivative(features,labels)
                        for layer in reversed(self.layers):
                                da = layer.backpropagate(da, self.learning_rate)
        
        def predict(self,data:np.ndarray)->List[int]:
                for layer in self.layers:
                        data = layer.forward_pass(data)
                return[0 if data[i]< 0.5 else 1 for i in range(data.size)]

# MSE loss der
def loss_derivative(expected, actual):
        return (expected-actual)

#BCE loss der
# def loss_derivative(expected, actual):
#         eps = 1e-15 
#         actual = np.clip(actual, eps, 1 - eps)
#         return (-expected / actual) + (1 - expected) / (1 - actual)

def relu(value):
        return np.maximum(0,value)

def relu_derivative(value):
        return value>0

def sigmoid(value):
        return 1/(1+np.exp(-value))

def sigmoid_derivative(value):
        return sigmoid(value)*(1-sigmoid(value))

training_spam = np.loadtxt(open("data/training_spam.csv"), delimiter=",")
testing_spam = np.loadtxt(open("data/testing_spam.csv"), delimiter=",")
labels = testing_spam[:,0]
tests = testing_spam[:,1:]

layers = [Layer(54,4,relu,relu_derivative),Layer(4,1,sigmoid,sigmoid_derivative)]
network = NeuralNetwork(layers,loss_derivative)

network.train(1000,training_spam)

predictions = network.predict(tests)
print(predictions, labels)
accuracy = np.count_nonzero(predictions == labels) * 100 / labels.shape[0]
print("\nAccuracy on test data is {:.1f}%.".format(accuracy))
