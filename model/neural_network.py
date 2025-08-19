from typing import Callable, List
import numpy.typing as npt
from model.layers import Layer

# multi-layer ANN
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

    def train(self, training_data: npt.NDArray, training_labels: npt.NDArray):
        # forward pass, note that input features become the predictions after the pass
        predictions = training_data
        for layer in self.layers:
            predictions = layer.forward_pass(predictions)

        # gradient of loss w.r.t output
        dA = self.loss_derivative(predictions, training_labels)

        # backpropagating each layer
        for layer in reversed(self.layers):
            dA = layer.backpropagate(dA, self.learning_rate)

    def predict(self, data: npt.NDArray) -> npt.NDArray:
        predictions = data
        for layer in self.layers:
            predictions = layer.forward_pass(predictions)
        return predictions
