import numpy as np
from typing import Callable, List
import numpy.typing as npt

# activation functions, loss functions, and derivatives


def relu(values: npt.NDArray) -> npt.NDArray:
    return np.maximum(0, values)


def relu_derivative(value: npt.NDArray) -> npt.NDArray:
    return value > 0


def sigmoid(value: npt.NDArray) -> npt.NDArray:
    return 1 / (1 + np.exp(-value))


def sigmoid_derivative(value: npt.NDArray) -> npt.NDArray:
    sigmoided = sigmoid(value)
    return sigmoided * (1 - sigmoided)


def MSE_loss_derivative(predictions: npt.NDArray, labels: npt.NDArray) -> npt.NDArray:
    return predictions - labels


def BCE_loss_derivative(predictions: npt.NDArray, labels: npt.NDArray) -> npt.NDArray:
    eps = 1e-15
    predictions = np.clip(predictions, eps, 1 - eps)
    return (-labels / predictions) + (1 - labels) / (1 - predictions)


# single layer of a neural network
class Layer:
    def __init__(
        self,
        input_size: int,
        num_neurons: int,
        activation_function: Callable[[npt.NDArray], npt.NDArray],
        activation_derivative: Callable[[npt.NDArray], npt.NDArray],
    ):
        self.weights = np.random.rand(input_size, num_neurons)
        self.biases = np.zeros((1, num_neurons))
        self.activation_function = activation_function
        self.activation_derivative = activation_derivative

        self.input_data = None
        self.forward = None
        self.activated = None

    def forward_pass(self, input_data: npt.NDArray) -> npt.NDArray:
        # basic batched y=mx+c with activation, y = act(mx+c)
        self.input_data = input_data
        self.forward = np.dot(input_data, self.weights) + self.biases
        self.activated = self.activation_function(self.forward)
        return self.activated

    def backpropagate(self, da: npt.NDArray, learning_rate: float) -> npt.NDArray:
        batch_size = da.shape[0]

        dZ = (
            self.activation_derivative(self.forward) * da
        )  # gradient w.r.t pre-activation
        dW = np.matmul(self.input_data.T, dZ) / batch_size  # gradient w.r.t weights
        dB = np.matmul(np.ones((batch_size, 1)).T, dZ) / batch_size  # gradient w.r.t bias

        # gradient update (basic for now)
        self.weights = self.weights - learning_rate * dW
        self.biases = self.biases - learning_rate * dB

        # sending gradient to previous layer
        next_da = np.dot(dZ, self.weights.T)
        return next_da


# multi-layer ann
class NeuralNetwork:
    def __init__(
        self,
        layers: List[Layer],
        loss_derivative: Callable[[npt.NDArray, npt.NDArray], npt.NDArray],
        learning_rate: float = 1.0,
    ):
        self.layers = layers
        self.learning_rate = learning_rate
        self.loss_derivative = loss_derivative

    def train(
        self, training_data: npt.NDArray, training_labels: npt.NDArray
    ):
        # forward pass
        features = training_data
        for layer in self.layers:
            features = layer.forward_pass(
                features
            )  # input features become outputs (predictions)

        # gradient of loss w.r.t output
        dA = self.loss_derivative(features, training_labels)

        # backpropagating each layer
        for layer in reversed(self.layers):
            dA = layer.backpropagate(dA, self.learning_rate)

    def predict(self, data: npt.NDArray) -> npt.NDArray:
        for layer in self.layers:
            data = layer.forward_pass(data)
        return data


training_spam = np.loadtxt(open("data/training_spam.csv"), delimiter=",")
train_labels = training_spam[:, 0][:, np.newaxis]
train_data = training_spam[:, 1:]

testing_spam = np.loadtxt(open("data/testing_spam.csv"), delimiter=",")
test_labels = testing_spam[:, 0]
test_data = testing_spam[:, 1:]

layers = [Layer(54, 4, relu, relu_derivative), Layer(4, 1, sigmoid, sigmoid_derivative)]
network = NeuralNetwork(layers, loss_derivative=BCE_loss_derivative, learning_rate=0.1)

epochs = 1000
for _ in range(epochs):
    network.train(train_data, train_labels)

predictions = network.predict(test_data).squeeze() > 0.5
accuracy = np.sum(predictions == test_labels) * 100 / test_labels.shape[0]
print("\nAccuracy on test data is {:.1f}%.".format(accuracy))
