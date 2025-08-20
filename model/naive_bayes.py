import numpy as np
import numpy.typing as npt

class NaiveBayes():
    def __init__(self, train_data: npt.NDArray, train_labels: npt.NDArray, smoothing: float = 1.0):
        self.smoothing = smoothing
        self.train_data = train_data
        self.train_labels = train_labels
        self.classes = None
        self.priors = {}
        self.likelihoods = {}
        self.prob_features = None
        self._postinit()

    def _postinit(self):
        self.classes = list(set(self.train_labels))
        # calculating the probability of features, even though its unused
        self.prob_features = np.sum(self.train_data, axis=0) / np.sum(self.train_data)
        self._calculate_priors()
        self._calculate_likelihoods()

    def _calculate_priors(self):
        total_training_samples = len(self.train_labels)
        for cls in self.classes:
            # number of samples of given class
            num_class_training_samples = np.sum(self.train_labels == cls)
            # prior probability for given class
            self.priors[cls] = num_class_training_samples / total_training_samples

    def _calculate_likelihoods(self):
        for cls in self.classes:
            # getting data from given class
            class_indexes = np.where(self.train_labels == cls)[0]
            class_training_data = self.train_data[class_indexes]

            # summing for each feature
            class_feature_instances = np.sum(class_training_data, axis=0)
            total_class_instances = len(class_indexes)
            # final likelihoods with Laplace smoothing
            self.likelihoods[cls] = (class_feature_instances + self.smoothing) / (total_class_instances + self.smoothing * len(self.train_data[0]))

    def predict_classes(self, test_data: npt.NDArray):
      #assumes test_data is 2d
        predictions = []
        for sample in test_data:
            class_probs = {}
            for cls in self.classes:
              # using log to avoid floating point precision error
                prior = np.log(self.priors[cls])
                # product becomes sum in log space
                likelihood = np.sum(np.log(self.likelihoods[cls])*sample)
                class_probs[cls] = prior + likelihood
            # maximum a posteriori, class with highest probability
            predictions.append(max(class_probs, key=class_probs.get))
        return predictions