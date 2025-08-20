import numpy as np
from model.layers import Layer
from model.neural_network import NeuralNetwork
import model.activations as act
import model.losses as loss
from nlp.utils import get_imdb_review_data
import nlp.preprocessing as prep
from train import train_model
import yaml

# import nltk
# nltk.download('averaged_perceptron_tagger_eng')
# nltk.download('stopwords')
# nltk.download('wordnet')

all_reviews, all_labels = get_imdb_review_data()
all_reviews, all_labels = all_reviews[:10000], all_labels[:10000]

with open("config.yaml", "r") as file:
    config_dict = yaml.safe_load(file)

yaml_preprocessing_config = config_dict["preprocessing"]
preprocessing_config = prep.PreprocessingConfig(
    tokeniser=prep.TokeniserType(yaml_preprocessing_config["tokeniser"]),
    normalisation=prep.NormalisationType(yaml_preprocessing_config["normalisation"]),
    keep_percent=yaml_preprocessing_config["keep_percent"],
    test_size=yaml_preprocessing_config["test_size"],
    val_size=yaml_preprocessing_config["val_size"],
    include_ngrams=yaml_preprocessing_config["include_ngrams"],
    ngram_n=yaml_preprocessing_config["ngram_n"],
    keep_ngrams=yaml_preprocessing_config["keep_ngrams"],
)

train_data, train_labels, val_data, val_labels, test_data, test_labels = (
    prep.preprocess_documents(all_reviews, all_labels, preprocessing_config)
)

layers = [
    Layer(train_data.shape[1], 50, act.relu, act.relu_derivative),
    Layer(50, 1, act.sigmoid, act.sigmoid_derivative),
]
network = NeuralNetwork(
    layers, loss_derivative=loss.BCE_loss_derivative, learning_rate=0.01
)

train_model(
    network,
    train_data,
    train_labels,
    val_data,
    val_labels,
    epochs=1000,
    batch_size=256,
    verbose=True,
)

predictions = network.predict(val_data).squeeze() > 0.5
accuracy = np.mean(predictions == val_labels) * 100
print("Accuracy on test data is {:.1f}%.".format(accuracy))

# could add more gradient update methods
