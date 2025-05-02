"""
Preprocess the data to be trained by the learning algorithm.
"""

import numpy as np
import pandas as pd
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_union, make_pipeline
from joblib import dump, load


def _load_data():
    reviews = pd.read_csv(
        'a1_RestaurantReviews_HistoricDump.tsv', 
        delimiter='\t', 
        quoting=3)
    return reviews

def _text_process(text):
    '''
    1. Remove all non-alphabetic characters
    2. Convert to lowercase
    3. Remove stopwords
    4. Stem the words
    '''
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
    # return as np.array and reshape so that it works with make_union
    return np.array([len(message) for message in data]).reshape(-1, 1)


def _preprocess(reviews):
    '''
    1. Convert
    '''
    preprocessor = make_union(
        CountVectorizer(max_features=1420,
                        analyzer=_text_process),
        FunctionTransformer(_extract_message_len, validate=False)
    )

    preprocessed_data = preprocessor.fit_transform(reviews['Review'])
    dump(preprocessor, 'Output/preprocessor.joblib')
    dump(preprocessed_data, 'Output/preprocessed_data.joblib')
    return preprocessed_data

def main():
    reviews = _load_data()
    _preprocess(reviews)

if __name__ == '__main__':
    main()