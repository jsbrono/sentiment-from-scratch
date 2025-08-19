import numpy as np
import numpy.typing as npt

def relu(values: npt.NDArray) -> npt.NDArray:
    return np.maximum(0, values)


def relu_derivative(value: npt.NDArray) -> npt.NDArray:
    return value > 0


def sigmoid(value: npt.NDArray) -> npt.NDArray:
    return 1 / (1 + np.exp(-value))


def sigmoid_derivative(value: npt.NDArray) -> npt.NDArray:
    sigmoided = sigmoid(value)
    return sigmoided * (1 - sigmoided)