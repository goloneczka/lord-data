from nltk.tokenize import sent_tokenize, word_tokenize

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import nltk

nltk.download('stopwords')
nltk.download('punkt')

LOTR_DATASETS = 'lotr_characters'
CLEAR_LOTR_DATASETS = 'cleaned_' + LOTR_DATASETS

stemmer = SnowballStemmer("english")
stopwords = stopwords.words('english')


def tokenize_and_stem(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if nltk.re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)

    # exclude stopwords from stemmed words
    stems = [stemmer.stem(t) for t in filtered_tokens if t not in stopwords]
    return stems


def get_dialogs_per_char(only_most_popular=False):
    most_popular_chars = ['frodo', 'sam', 'gandalf', 'aragorn', 'pippin', 'merry', 'gollum', 'gimli', 'theoden',
                          'faramir']
    dictionary_chars = {}
    csv_reader = pd.read_csv("./" + CLEAR_LOTR_DATASETS + '/' + 'lotr_scripts.csv', usecols=['char', 'dialog'])
    if only_most_popular:
        for _, value in csv_reader.iterrows():
            if (value.char not in dictionary_chars) and (value.char in most_popular_chars):
                dictionary_chars[value.char] = [value.dialog]
            elif value.char in most_popular_chars:
                dictionary_chars[value.char].append(value.dialog)
    else:
        for _, value in csv_reader.iterrows():
            if value.char not in dictionary_chars:
                dictionary_chars[value.char] = [value.dialog]
            else:
                dictionary_chars[value.char].append(value.dialog)

    corpus = []
    for _, value in dictionary_chars.items():
        corpus.append(' '.join(map(str, value)))

    return corpus


def vectorize_dialogs(only_most_popular=False):
    corpus = get_dialogs_per_char(only_most_popular)
    tv = TfidfVectorizer(binary=False, norm=None, use_idf=False,
                         smooth_idf=False, lowercase=True, stop_words='english',
                         min_df=1, max_df=1.0, max_features=None, tokenizer=tokenize_and_stem,
                         ngram_range=(3, 3))

    df = pd.DataFrame(tv.fit_transform(corpus).toarray(), columns=tv.get_feature_names_out())
    return df


def get_most_popular_phrase_by_char():
    # znam kolejnosc przez printa dictionary_chars z poprzedniej metody
    most_popular_sorted_chars = ['frodo', 'merry', 'gimli', 'gollum', 'sam', 'gandalf', 'aragorn', 'pippin', 'theoden',
                                 'faramir']
    dictionary = {}

    df = vectorize_dialogs(True)

    df['max_value'] = df.max(axis=1)
    df['most_popular_sequence'] = df.idxmax(axis=1)

    popular_df = df.filter(['max_value', 'most_popular_sequence'], axis=1).copy()
    popular_df['char'] = most_popular_sorted_chars

    return popular_df
