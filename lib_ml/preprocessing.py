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
nltk.download('stopwords')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_union
from joblib import dump


def _load_data(path: Path):
    reviews = pd.read_csv(
        path, 
        delimiter='\t', 
        quoting=3)
    return reviews

def _text_process(text: str):
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    ps = PorterStemmer()

    clean_rvws = []

    for i in range(0, 900):
        review = re.sub('[^a-zA-Z]', ' ', text)
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
        review = ' '.join(review)
        clean_rvws.append(review)
    
    return clean_rvws

def _extract_message_len(data):
    return np.array([len(message) for message in data]).reshape(-1, 1)


def preprocess(path: Path):
    reviews = _load_data(path)

    preprocessor = make_union(
        CountVectorizer(max_features=1420,
                        analyzer=_text_process),
        FunctionTransformer(_extract_message_len, validate=False)
    )

    preprocessed_data = preprocessor.fit_transform(reviews['Review'])
    dump(preprocessor, 'Output/preprocessor.joblib')
    dump(preprocessed_data, 'Output/preprocessed_data.joblib')
    return preprocessed_data
