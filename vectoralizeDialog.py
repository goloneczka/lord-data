from nltk.tokenize import sent_tokenize, word_tokenize

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import nltk
from textblob import TextBlob
from sklearn import cluster

nltk.download('stopwords')
nltk.download('punkt')

LOTR_DATASETS = 'lotr_characters'
CLEAR_LOTR_DATASETS = 'cleaned_' + LOTR_DATASETS

stemmer = SnowballStemmer("english")
stopwords = stopwords.words('english')
stopwords.remove('my')

# znam kolejnosc przez printa dictionary_chars z metody get_dialogs_per_char
most_popular_sorted_chars = ['gollum', 'frodo', 'merry', 'gimli', 'sam', 'gandalf', 'aragorn', 'pippin', 'theoden',
                            'faramir']
# most_popular_sorted_chars = ['frodo', 'merry', 'sam', 'pippin', 'aragorn', 'theoden', 'faramir', 'boromir', 'gollum', 'gimli']

def tokenize_and_stem(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if nltk.re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)

    # exclude stopwords from stemmed words
    stems = [stemmer.stem(t) for t in filtered_tokens if t not in stopwords]

    return stems


def get_dialogs_per_char(only_most_popular=True):
    dictionary_chars = {}
    csv_reader = pd.read_csv("./" + CLEAR_LOTR_DATASETS + '/' + 'lotr_scripts.csv', usecols=['char', 'dialog'])
    if only_most_popular:
        for _, value in csv_reader.iterrows():
            if (value.char not in dictionary_chars) and (value.char in most_popular_sorted_chars):
                dictionary_chars[value.char] = [value.dialog]
            elif value.char in most_popular_sorted_chars:
                dictionary_chars[value.char].append(value.dialog)
    else:
        for _, value in csv_reader.iterrows():
            if value.char not in dictionary_chars:
                dictionary_chars[value.char] = [value.dialog]
            else:
                dictionary_chars[value.char].append(value.dialog)

    return dictionary_chars


def get_vectorizer():
    return TfidfVectorizer(lowercase=True, stop_words=stopwords, tokenizer=tokenize_and_stem)

def vectorize_dialogs(only_most_popular=True):
    dictionary_chars = get_dialogs_per_char(only_most_popular)
    corpus = []
    for _, value in dictionary_chars.items():
        corpus.append(' '.join(map(str, value)))
    tv = get_vectorizer()
    tv.fit_transform(corpus)
    df = pd.DataFrame(tv.fit_transform(corpus).toarray(), columns=tv.get_feature_names_out())
    return df, tv


def get_most_popular_phrase_by_char():
    df, tv = vectorize_dialogs(True)

    df['max_value'] = df.max(axis=1)
    df['most_popular_sequence'] = df.idxmax(axis=1)

    popular_df = df.filter(['max_value', 'most_popular_sequence'], axis=1).copy()
    popular_df['char'] = most_popular_sorted_chars

    return popular_df


def get_dialog_sentiment():
    dictionary = {}
    indx = 0
    for key, value in get_dialogs_per_char(True).items():
        positive = 0
        negative = 0
        neutral = 0
        for dialog in value:
            if isinstance(dialog, str):
                sentiment = TextBlob(dialog).sentiment
                if sentiment.polarity > 0.3:
                    positive += 1
                elif sentiment.polarity < -0.3:
                    negative += 1
                else:
                    neutral += 1
        dictionary[most_popular_sorted_chars[indx]] = {'positive': positive, 'neutral': neutral, 'negative': negative}
        indx += 1

    return  dictionary


def draw_sentiment_diagram(type):
    dialog_dictionary = get_dialog_sentiment()
    dictionary_to_draw = {}
    for key, value in dialog_dictionary.items():
        dictionary_to_draw[key] = value[type] / (value['positive'] + value['neutral'] + value['negative'])

    return {k: v for k, v in sorted(dictionary_to_draw.items(), key=lambda item: item[1], reverse=True)}
