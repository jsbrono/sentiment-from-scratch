from typing import List,Tuple,Dict
from collections import Counter

def extract_ngrams(processed_documents: List[List[str]], n: int)->Tuple[Dict[str, int], List[List[str]]]:
  """Extracts n-grams from tokenised documents and appends them to each documents. Both done in a single pass to avoid
    additional looping later, in the case where you need the documents with the n-grams.

  Args:
      processed_documents (List[List[str]]): List of tokenised documents (each document being a list of token)
      n (int): the number of tokens per n-gram

  Returns:
      Tuple[Dict[str, int], List[List[str]]]: Tuple containing:
      - A dictionary of n-gram frequencies sorted in descending order, e.g.: "good movie": 32 for n=2
      - A list of documents where n-grams have been appended, e.g. "good","movie","good movie"
  """
  ngram_frequencies = Counter()
  documents_with_ngrams = []

  for document in processed_documents:
      document_copy = document.copy()
      # making the n-grams
      ngrams = zip(*[document[i:] for i in range(n)])
      for ngram in ngrams:
          joined_ngram = ' '.join(ngram)
          ngram_frequencies.update([joined_ngram])
          document_copy.append(joined_ngram) # appending ngrams to each document
      documents_with_ngrams.append(document_copy)

  sorted_frequencies = dict(sorted(ngram_frequencies.items(), key=lambda x: x[1], reverse=True))
  return sorted_frequencies,documents_with_ngrams

