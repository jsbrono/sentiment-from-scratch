from model.metrics import accuracy,precision,recall,f1_score
import numpy.typing as npt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Optional
import numpy as np

def evaluate_model(model,test_data:npt.NDArray,test_labels:npt.NDArray):
    predictions = model.predict(test_data).squeeze()>0.5

    acc = accuracy(test_labels, predictions)
    prec = precision(test_labels, predictions)
    rec = recall(test_labels, predictions)

    print(f"Accuracy: {acc:.1f}%")
    print(f"Precision: {prec:.2f}")
    print(f"Recall: {rec:.2f}")

def plot_1D(y_data:npt.NDArray,x_data:Optional[npt.NDArray]=None,title:str = "",x_label:str="",y_label:str="",save_filepath:Optional[str]=None):
    # Assume if 2D that we are plotting std as well, and that first dimension is number of "runs"

    if y_data.ndim>2:
        raise ValueError("y_data must be 1D or 2D")
    
    if x_data is None:
        x_data = np.arange(y_data.shape[-1])

    if y_data.ndim == 2:
        y_std = np.std(y_data,axis=0)
        y_mean = np.mean(y_data,axis=0) # plot the mean as the line instead

    plt.figure()
    if y_data.ndim == 2:
        plt.plot(x_data,y_mean,label="Mean")
        plt.fill_between(x_data,y_mean - y_std,y_mean+y_std,alpha=0.2,label = "Std")
    else:
        plt.plot(x_data,y_data)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_filepath)

