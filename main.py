import numpy as np
from model.layers import Layer
from model.neural_network import NeuralNetwork
import model.activations as act
import model.losses as loss

training_spam = np.loadtxt(open("data/training_spam.csv"), delimiter=",")
train_labels = training_spam[:, 0][:, np.newaxis]
train_data = training_spam[:, 1:]

testing_spam = np.loadtxt(open("data/testing_spam.csv"), delimiter=",")
test_labels = testing_spam[:, 0]
test_data = testing_spam[:, 1:]

layers = [Layer(54, 4, act.relu, act.relu_derivative), Layer(4, 1, act.sigmoid, act.sigmoid_derivative)]
network = NeuralNetwork(layers, loss_derivative=loss.BCE_loss_derivative, learning_rate=0.1)

epochs = 1000
for _ in range(epochs):
    network.train(train_data, train_labels)

predictions = network.predict(test_data).squeeze() > 0.5
accuracy = np.sum(predictions == test_labels) * 100 / test_labels.shape[0]
print("Accuracy on test data is {:.1f}%.".format(accuracy))

#  different gradient update methods etc
# config.yaml
