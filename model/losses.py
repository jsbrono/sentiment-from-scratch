import numpy as np
import numpy.typing as npt

def MSE_loss_derivative(predictions: npt.NDArray, labels: npt.NDArray) -> npt.NDArray:
    return predictions - labels


def BCE_loss_derivative(predictions: npt.NDArray, labels: npt.NDArray) -> npt.NDArray:
    eps = 1e-9
    predictions = np.clip(predictions, eps, 1 - eps)
    return (-labels / predictions) + (1 - labels) / (1 - predictions)