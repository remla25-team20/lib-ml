"""
Preprocess the data to be trained by the learning algorithm.
"""

import numpy as np
import pandas as pd
import re

from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from joblib import dump

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

def _load_data(path: Path):
    reviews = pd.read_csv(
        path, 
        delimiter='\t', 
        quoting=3)
    return reviews

def _text_process(text: str):
    all_stopwords = stopwords.words('english')
    for stopword in ['not', 'but']:
        all_stopwords.remove(stopword)
    ps = PorterStemmer()
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    return review

def preprocess(path: Path):
    reviews = _load_data(path)

    preprocessor = CountVectorizer(max_features=1420,
                        analyzer=_text_process)

    preprocessed_data = preprocessor.fit_transform(reviews['Review'])
    return preprocessor, preprocessed_data
