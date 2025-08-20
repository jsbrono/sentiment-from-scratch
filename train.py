import numpy as np
import numpy.typing as npt
from typing import Optional

def train_model(model,train_data:npt.NDArray,train_labels:npt.NDArray,val_data:Optional[npt.NDArray]=None,val_labels:Optional[npt.NDArray]=None,epochs:int=1000,batch_size:int=256,verbose:bool=False):
    for epoch in range(epochs):
        #shuffling 
        indices = np.random.permutation(train_data.shape[0])
        shuffled_reviews = train_data[indices]
        shuffled_labels = train_labels[:, np.newaxis][indices]

        #batching
        for start in range(0, shuffled_reviews.shape[0], batch_size):
            end = start + batch_size
            model.train(shuffled_reviews[start:end], shuffled_labels[start:end])

        if epoch % 10 == 0 and verbose and val_data is not None:
            predictions = model.predict(val_data).squeeze() > 0.5
            accuracy = np.mean(predictions == val_labels) * 100
            print(f"Epoch {epoch}, validation accuracy: {accuracy:.2f}%")
