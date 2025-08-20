import numpy as np
from typing import Callable
import numpy.typing as npt

# single layer of a neural network
class Layer:
    def __init__(
        self,
        input_size: int,
        num_neurons: int,
        activation_function: Callable[[npt.NDArray], npt.NDArray],
        activation_derivative: Callable[[npt.NDArray], npt.NDArray],
    ):
        self.weights = np.random.randn(input_size, num_neurons) * np.sqrt(2/input_size) # he initialisation, assume relu everywhere for now
        self.biases = np.zeros((1, num_neurons))
        self.activation_function = activation_function
        self.activation_derivative = activation_derivative

        # populated in forward pass
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
        dB = (
            np.matmul(np.ones((batch_size, 1)).T, dZ) / batch_size
        )  # gradient w.r.t bias

        # gradient update (basic for now)
        self.weights = self.weights - learning_rate * dW
        self.biases = self.biases - learning_rate * dB

        # sending gradient to previous layer
        next_da = np.dot(dZ, self.weights.T)
        return next_da