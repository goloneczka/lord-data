from nltk.tokenize import sent_tokenize, word_tokenize

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import PorterStemmer

import nltk



LOTR_DATASETS = 'lotr_characters'
CLEAR_LOTR_DATASETS = 'cleaned_' + LOTR_DATASETS

nltk.download('stopwords')
nltk.download('punkt')
# STOPWORDS = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")
stopwords = stopwords.words('english')

porter=PorterStemmer()


def stemSentence(sentence):
    token_words=word_tokenize(sentence)
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)

def stemming_tokenizer(str_input):
    words = nltk.re.sub(r"[^A-Za-z0-9\-]", " ", str_input).lower().split()
    words = [porter.stem(word) for word in words]
    return words

def tokenize_and_stem(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if nltk.re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)

    #exclude stopwords from stemmed words
    stems = [stemmer.stem(t) for t in filtered_tokens if t not in stopwords]

    return stems

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
                         min_df=1, max_df=1.0, max_features=None, tokenizer=tokenize_and_stem,
                         ngram_range=(1,1))

    tv._validate_vocabulary()
    df = pd.DataFrame(tv.fit_transform(corpus).toarray(), columns=tv.get_feature_names_out())
    for col in tv.get_feature_names_out():
        print(col)
    print(len(tv.get_feature_names_out()))
    print(df)

