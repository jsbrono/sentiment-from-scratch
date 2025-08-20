import numpy as np
from typing import List, Dict, Tuple, Any
from nltk.stem import WordNetLemmatizer, LancasterStemmer
import re
from nltk.corpus import stopwords
from collections import defaultdict
import numpy.typing as npt
from sklearn.model_selection import train_test_split
from nlp.utils import make_frequency_map, word_count_matrix
from enum import Enum
from dataclasses import dataclass
from nlp.feature_engineering import extract_ngrams


class TokeniserType(Enum):
    SPLIT = "split"
    STEM = "stem"
    LEMMATIZE = "lemmatize"


class NormalisationType(Enum):
    L2 = "L2"
    TFIDF = "tfidf"


@dataclass
class PreprocessingConfig:
    tokeniser: TokeniserType
    normalisation: NormalisationType
    keep_percent: float = 0.9
    test_size: float = 0.15
    val_size: float = 0.15
    include_ngrams: bool = False
    ngram_n: int = 2
    keep_ngrams: int = 1000


def split_by_whitespace(documents: List[str]) -> List[List[str]]:
    return np.concatenate([doc.split() for doc in documents])


def split_and_remove_punctuation(documents: List[str]) -> List[List[str]]:
    """Splits text and removes punctuation, common HTML tokens, and stopwords.

    Args:
        reviews (List[str]): List of unsplit documents

    Returns:
        List[List[str]]: Split documents into tokens
    """
    stop_words = set(stopwords.words("english"))
    return [
        [
            word
            for word in re.sub(r"(?:br|[^\w\s])", "", doc).split()
            if word.lower() not in stop_words
        ]
        for doc in documents
    ]


def lemmatize(reviews: List[str]) -> List[List[str]]:
    reviews = split_and_remove_punctuation(reviews)
    lemmatizer = WordNetLemmatizer()
    return [[lemmatizer.lemmatize(word) for word in words] for words in reviews]


def stem(reviews: List[str]) -> List[List[str]]:
    reviews = split_and_remove_punctuation(reviews)
    stemmer = LancasterStemmer()
    return [[stemmer.stem(word) for word in words] for words in reviews]


def filter_by_frequency(
    frequency_map: Dict[Any, int], keep: int, most_frequent: bool = True
) -> Tuple[Dict[Any, int], Dict[Any, int]]:
    """Filters a frequency map, used to remove most or least frequent occurrences in the map.

    Args:
        frequency_map (Dict[Any,int]): The frequency dictionary.
        keep (int): The number of items to keep.
        most_frequent (bool, optional): If set to True, keeps the most frequent. Otherwise filters out least frequent. Defaults to True.

    Returns:
        Tuple[Dict[Any,int],Dict[Any,int]]: Tuple containing:
        - Dictionary of kept items.
        - Dictionary of removed items.
    """
    frequencies = sorted(frequency_map.items(), key=lambda x: x[1], reverse=True)
    if not most_frequent:
        frequencies = reversed(frequencies)
    kept_items = dict(frequencies[:keep])
    removed_items = dict(frequencies[keep:])

    return kept_items, removed_items


def calculate_tfidf(
    reviews: List[List[str]], vocabulary: List[str]
) -> tuple[npt.NDArray, npt.NDArray]:
    vocabulary_index_dict = {word: i for i, word in enumerate(vocabulary)}

    # creating matrix
    review_tf_matrix = np.zeros((len(reviews), len(vocabulary)))

    # number of reviews where word appears, used for idf later
    num_reviews_with_word = np.zeros(len(vocabulary))

    # getting word counts
    for i, review in enumerate(reviews):
        # getting word counts in a review
        review_word_counts = defaultdict(int)
        for word in review:
            review_word_counts[word] += 1

        # updating the matrix
        for word, wordcount in review_word_counts.items():
            if word in vocabulary_index_dict:
                review_tf_matrix[i][vocabulary_index_dict[word]] += wordcount

                # updating global word count
                num_reviews_with_word[vocabulary_index_dict[word]] += 1

    # final tf values
    for i in range(len(review_tf_matrix)):
        review_tf_matrix[i] /= np.sum(review_tf_matrix[i])

    # calculating idf
    total_reviews = len(reviews)
    review_idfs = np.array([1 + total_reviews] * len(vocabulary))
    review_idfs = np.log(review_idfs / (1 + num_reviews_with_word))
    # final tfidf values
    tfidfs = review_tf_matrix * review_idfs
    # return review_idfs to use for evaluation/testing
    return tfidfs, review_idfs


def l2_norm(reviews: List[List[str]], vocabulary: List[str]) -> npt.NDArray:
    vocabulary_index_dict = {word: i for i, word in enumerate(vocabulary)}
    # initialising word count matrix
    review_word_count_matrix = np.zeros((len(reviews), len(vocabulary)))

    for i, review in enumerate(reviews):
        # getting word counts
        review_word_counts = defaultdict(int)
        for word in review:
            review_word_counts[word] += 1

        # updating word counts
        for word, wordcount in review_word_counts.items():
            review_word_count_matrix[i][vocabulary_index_dict[word]] += wordcount

    # normalising word counts
    sums = np.sum(review_word_count_matrix**2, axis=1)
    norm_factor = np.sqrt(sums)
    l2_normed = review_word_count_matrix / norm_factor[:, None]

    return l2_normed


def preprocess_documents(
    documents: npt.NDArray, labels: npt.NDArray, config: PreprocessingConfig
) -> Tuple[
    npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray
]:

    # Tokenising
    if config.tokeniser == TokeniserType.SPLIT:
        tokenised_documents = split_and_remove_punctuation(documents)
    elif config.tokeniser == TokeniserType.STEM:
        tokenised_documents = stem(documents)
    elif config.tokeniser == TokeniserType.LEMMATIZE:
        tokenised_documents = lemmatize(documents)
    else:
        raise ValueError("Unknown tokeniser")

    # Data split
    train_documents, test_val_data, train_labels, test_val_labels = train_test_split(
        tokenised_documents,
        labels,
        test_size=(config.val_size + config.test_size),
        random_state=1,
    )
    relative_test_size = config.test_size / (config.test_size + config.val_size)
    val_documents, test_documents, val_labels, test_labels = train_test_split(
        test_val_data, test_val_labels, test_size=relative_test_size, random_state=1
    )

    # Token filtering
    token_frequencies_train = make_frequency_map(train_documents)
    num_tokens_to_keep = int(len(token_frequencies_train) * config.keep_percent)
    kept_tokens, _ = filter_by_frequency(
        token_frequencies_train, num_tokens_to_keep, True
    )

    # Optional n-grams
    kept_ngrams = {}
    if config.include_ngrams:
        ngram_train_frequencies, documents_with_ngrams_train = extract_ngrams(
            train_documents, n=config.ngram_n
        )
        _, documents_with_ngrams_val = extract_ngrams(val_documents, n=config.ngram_n)
        _, documents_with_ngrams_test = extract_ngrams(test_documents, n=config.ngram_n)
        kept_ngrams, _ = filter_by_frequency(
            ngram_train_frequencies, config.keep_ngrams, most_frequent=True
        )
        train_documents = documents_with_ngrams_train
        val_documents = documents_with_ngrams_val
        test_documents = documents_with_ngrams_test

    # Token removal
    to_keep = set(list(kept_tokens.keys()) + list(kept_ngrams.keys()))

    shortened_documents_train = [
        [token for token in document if token in to_keep]
        for document in train_documents
    ]
    shortened_documents_val = [
        [token for token in document if token in to_keep] for document in val_documents
    ]
    shortened_documents_test = [
        [token for token in document if token in to_keep] for document in test_documents
    ]

    # Normalisation

    vocabulary = list(set(np.concatenate(shortened_documents_train)))

    if config.normalisation == NormalisationType.TFIDF:
        normalised_reviews_train, train_idfs = calculate_tfidf(
            shortened_documents_train, vocabulary
        )
        val_tfs = word_count_matrix(
            shortened_documents_val, {token: i for i, token in enumerate(vocabulary)}
        )
        test_tfs = word_count_matrix(
            shortened_documents_val, {token: i for i, token in enumerate(vocabulary)}
        )
        normalised_reviews_val = val_tfs * train_idfs
        normalised_reviews_test = test_tfs * train_idfs
    elif config.normalisation == NormalisationType.L2:
        normalised_reviews_train = l2_norm(shortened_documents_train, vocabulary)
        normalised_reviews_val = l2_norm(shortened_documents_val, vocabulary)
        normalised_reviews_test = l2_norm(shortened_documents_test, vocabulary)

    return (
        normalised_reviews_train,
        train_labels,
        normalised_reviews_val,
        val_labels,
        normalised_reviews_test,
        test_labels,
    )
