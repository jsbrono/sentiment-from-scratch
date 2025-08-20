import numpy as np
from typing import List,Dict
from collections import defaultdict
import numpy.typing as npt
import kagglehub 
import pandas as pd

def word_count_matrix(reviews:List[List[str]],vocabulary_index_dict:Dict[str,int])->npt.NDArray:
  review_word_count_matrix = np.zeros((len(reviews),len(vocabulary_index_dict)))
  for i,review in enumerate(reviews):
    for word in review:
          review_word_count_matrix[i][vocabulary_index_dict[word]] += 1

  return review_word_count_matrix

def make_frequency_map(reviews:List[List[str]])->Dict[str,int]:
    #assumes split
    freqs = defaultdict(int)
    for review in reviews:
        for word in review:
            freqs[word]+=1
    return dict(sorted(freqs.items(),key=lambda x: x[1], reverse=True))

def get_imdb_review_data():
    path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
    csv_path = path + "/IMDB Dataset.csv"
    df = pd.read_csv(csv_path)
    df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})

    reviews = df["review"].to_numpy()
    labels = df["sentiment"].to_numpy()

    return reviews,labels
