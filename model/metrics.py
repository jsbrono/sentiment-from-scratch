import numpy as np
import numpy.typing as npt

def accuracy(predictions:npt.NDArray, labels:npt.NDArray, percent:bool=False)->float:
    if percent == True:
        return np.mean(predictions == labels) * 100 
    return np.mean(predictions == labels)

def precision(predictions:npt.NDArray, labels:npt.NDArray)->float:
    true_positives = np.sum((predictions == 1) & (labels==1))
    false_positives = np.sum((predictions==1) & (labels==0))
    return true_positives / (true_positives+false_positives)

def recall(predictions:npt.NDArray, labels:npt.NDArray)->float:
    true_positives = np.sum((predictions == 1) & (labels==1))
    false_negatives = np.sum((predictions == 0) & (labels==1))
    return true_positives / (true_positives + false_negatives + 1e-9) # avoid /0 error

def f1_score(predictions:npt.NDArray, labels:npt.NDArray)->float:
    prec = precision(predictions,labels)
    rec = recall(predictions,labels)
    return 2*prec*rec / (prec+rec+1e-9) # avoid /0 error