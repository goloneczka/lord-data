import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import nltk

from nltk.tokenize import TweetTokenizer, sent_tokenize


LOTR_DATASETS = 'lotr_characters'
CLEAR_LOTR_DATASETS = 'cleaned_' + LOTR_DATASETS

nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")


def get_dialogs_per_char():
    dictionary_chars = {}
    csv_reader = pd.read_csv("./" + CLEAR_LOTR_DATASETS + '/' + 'lotr_scripts.csv', usecols=['char', 'dialog'])
    for _, value in csv_reader.iterrows():
        if value.char not in dictionary_chars:
            dictionary_chars[value.char] = [value.dialog]
        else:
            dictionary_chars[value.char].append(value.dialog)

    corpus = []
    for _, value in dictionary_chars.items():
        corpus.append(' '.join(map(str, value)))

    tv = TfidfVectorizer(binary=False, norm=None, use_idf=False,
                         smooth_idf=False, lowercase=True, stop_words='english',
                         min_df=1, max_df=1.0, max_features=None,
                         ngram_range=(1,1))

    df = pd.DataFrame(tv.fit_transform(corpus).toarray(), columns=tv.get_feature_names_out())
    for col in tv.get_feature_names_out():
        print(col)
    print(len(tv.get_feature_names_out()))

