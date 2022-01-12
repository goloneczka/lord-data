#!/usr/bin/python3

import os
import re

from staticAnalize import *
from vectoralizeDialog import *
from kmeans import *

from sklearn.metrics import silhouette_score

os.environ['KAGGLE_USERNAME'] = "michalmichael"
os.environ['KAGGLE_KEY'] = "212e4a92b4a5f6143a6a3fc26c2375bd"

import kaggle
import enums

LOTR_DATASETS = 'lotr_characters'
CLEAR_LOTR_DATASETS = 'cleaned_' + LOTR_DATASETS

# Classification
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow import string as tf_string
from keras.models import Model
from tensorflow.keras.layers import TextVectorization
from keras.layers import Input, Embedding, Dropout, Dense, LSTM
from keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical


def get_kaggle_data():
    try:
        os.mkdir(LOTR_DATASETS)
    except OSError:
        print("Structured dir created - skip structuring data")
        return

    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('paultimothymooney/lord-of-the-rings-data', path=LOTR_DATASETS,
                                      unzip=True)


def clear_data():
    try:
        os.mkdir(CLEAR_LOTR_DATASETS)
    except OSError:
        print("cleaned dir created - skip cleaning data")
        return

    csv_reader = pd.read_csv("./" + LOTR_DATASETS + '/' + 'lotr_characters.csv')
    for _, val in csv_reader.iterrows():
        # val['gender'] = 'UNKNOWN' if not isinstance(val['gender'], str) else val['gender']
        val["name"] = val["name"].lower()
        val["name"] = enums.NAME_DICTIONARY[val["name"]] if val["name"] in enums.NAME_DICTIONARY else val["name"]
        val["race"] = enums.RACE_DICTIONARY[val["race"]] if val["race"] in enums.RACE_DICTIONARY else val["race"]

    csv_reader.to_csv("./" + CLEAR_LOTR_DATASETS + '/' + 'lotr_characters.csv')

    csv_reader = pd.read_csv("./" + LOTR_DATASETS + '/' + 'lotr_scripts.csv')
    del csv_reader['Unnamed: 0']
    for _, val in csv_reader.iterrows():
        val["char"] = val["char"].lower()
        val["char"] = enums.NAME_DICTIONARY[val["char"]] if val["char"] in enums.NAME_DICTIONARY else val["char"]
        val["dialog"] = val["dialog"] if isinstance(val["dialog"], str) else " "
        val["dialog"] = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", " ", val["dialog"])
        val["dialog"] = " ".join(val["dialog"].split()).lower()

    csv_reader.to_csv("./" + CLEAR_LOTR_DATASETS + '/' + 'lotr_scripts.csv')

characters_names = [
    'deagol',
    'smeagol',
    'gollum',
    'frodo',
    'merry',
    'gimli',
    'sam',
    'gandalf',
    'aragorn',
    'pippin',
    'hobbit',
    'rosie',
    'bilbo',
    'saruman',
    'theoden',
    'galadril',
    'elrond',
    'grima',
    'witch king',
    'eowyn',
    'faramir',
    'orc',
    'soldiers gate',
    'gothmog',
    'general',
    'captain',
    'soldier',
    'sauron',
    'eomer',
    'army',
    'boson',
    'mercenary',
    'eowyn merry',
    'denethor',
    'rohirrim',
    'galadriel',
    'legolas',
    'king dead',
    'grimbold',
    'irolas',
    'orcs',
    'gamling',
    'madril',
    'damrod',
    'soldiers',
    'soldiers minas tirith',
    'woman',
    'haldir',
    'old man',
    'boromir',
    'crowd',
    'arwen',
    'hama',
    'sharku',
    'people',
    'lady',
    'freda',
    'morwen',
    'rohan stableman',
    'gorbag',
    'ugluk',
    'shagrat',
    'uruk hai',
    'snaga',
    'grishnakh',
    'merry pippin',
    'wildman',
    'strider',
    'eothain',
    'rohan horseman',
    'farmer maggot',
    'white wizard',
    'gaffer',
    'noakes',
    'sandyman',
    'figwit',
    'general shout',
    'grishnak',
    'mrs bracegirdle',
    'proudfoot hobbit',
    'gatekeepr',
    'man',
    'children hobbits',
    'barliman',
    'ring',
    'men']

races = [
    'stoor' ,
    'stoor',
    'stoor',
    'hobbit',
    'hobbit',
    'dwarf',
    'hobbit',
    'maia',
    'human',
    'hobbit',
    'hobbit',
    'hobbit',
    'hobbit',
    'maia',
    'human',
    'elf',
    'elf',
    'human',
    'ringwraith',
    'human',
    'human',
    'orc',
    'human',
    'orc',
    'orc',
    'human',
    'human',
    'maia' ,
    'human',
    'human',
    'boson',
    'human',
    'human',
    'human',
    'human',
    'elf',
    'elf',
    'ringwraith',
    'human',
    'elf',
    'orc',
    'human' ,
    'human',
    'human',
    'human',
    'human',
    'human',
    'elf',
    'human',
    'human',
    'human',
    'elf',
    'human',
    'orc',
    'human',
    'human',
    'human',
    'human',
    'human',
    'orc',
    'orc',
    'orc',
    'orc',
    'orc',
    'orc',
    'hobbit' ,
    'human',
    'human',
    'human',
    'human',
    'human',
    'maia',
    'hobbit',
    'hobbit',
    'hobbit',
    'elf',
    'human',
    'orc',
    'hobbit',
    'hobbit',
    'human',
    'human',
    'hobbit' ,
    'human',
    'ringwraith',
    'human']

def get_dialogs(label_col):
    all_dialogs = []
    labels = []
    characters = get_characters_metadata()
    dialogs = get_dialogs_per_char()
    for character_name in characters_names:
        if character_name not in dialogs:
            continue
        for dialog in dialogs[character_name]:
            if character_name in characters:
                all_dialogs.append(dialog)
                labels.append(characters[character_name][label_col])
    return all_dialogs, labels

def class_report(y_test, y_pred_vect):
    y_pred = np.argmax(y_pred_vect, axis=1)
    print(classification_report(y_true=y_test, y_pred=y_pred))

    conf_mtx = confusion_matrix(y_test, y_pred)
    races = ['dwarf',
             'elf',
             'ent',
             'hobbit',
             'maia',
             'hobbit',
             'human',
             'ringwraith',
             'stoor']
    df_conf_mtx = pd.DataFrame(conf_mtx, index=races, columns=races)
    plt.figure(figsize=(12, 5))
    sns.heatmap(df_conf_mtx, fmt='d', annot=True, cmap='Reds')
    plt.xlabel('Predicted label', size = 15)
    plt.ylabel('True label', size= 15)
    plt.title('Confusion matrix', size=20)
    plt.show()

if __name__ == "__main__":
    get_kaggle_data()
    clear_data()

    # print(get_dialog_sentiment())
    # draw_histogram_by_dictionary(draw_sentiment_diagram('positive'), 'percentage of positive dialogs', 'value', 'char')
    # draw_histogram_by_dictionary(draw_sentiment_diagram('neutral'), 'percentage of neutral dialogs', 'value', 'char')
    # draw_histogram_by_dictionary(draw_sentiment_diagram('negative'), 'percentage of negative dialogs', 'value', 'char')
    #
    #
    # print(get_most_popular_phrase_by_char())

    data, vectorizer = vectorize_dialogs(True)

    # kmeans_results = run_KMeans(5, data)
    # kmeans = kmeans_results.get(2)

    db_scan = run_DBSan(data)
    final_df_array = data.to_numpy()
    # n_feats = 10
    # dfs = get_top_features_cluster(final_df_array, db_scan, n_feats, vectorizer)
    # plotWords(dfs, 13)

    print(silhouette_score(final_df_array, db_scan))

    # if you want to see heroes assigned to clusters comment out lines below and comment the ones above
    # labels = db_scan.labels_
    # chars = pd.DataFrame(get_dialogs_per_char(True).keys())
    #
    # chars['label'] = labels
    # print(chars)

    #  ---- STATIC ANALIZE -- task 1
    # print(count_dialogs_by_race())
    # draw_histogram_by_dictionary(count_dialogs_by_race(), 'amount of dialogs by race', 'value', 'race')
    # print(count_races_by_heroes())
    # print(count_dialogs_by_movie())
    # draw_histogram_by_dictionary(count_races_by_heroes(), 'amount of heroes in races', 'value', 'race')
    # draw_histogram_by_dictionary(count_dialogs_by_movie(), 'amount of dialogs per movie', 'value', 'movie')
    # draw_pie_chart(count_heroes_by_gender())
    # print(count_heroes_by_dialogs())
    # draw_histogram_by_dictionary(count_heroes_by_dialogs(), 'amount of dialogs by hero', 'value', 'hero')
    # print(average_dialogs_length_by_hero())
    # draw_histogram_by_dictionary(average_dialogs_length_by_hero(), 'average length of sentence by hero', 'value',
    #                              'hero')
    # print(count_gender_dialogs_by_move())
    # draw_histogram_by_dictionary_with_dictionary(count_gender_dialogs_by_move(), 'sentences by gender in movies',
    #                                              'value', 'movie')
    #
    # race_dialogs_by_movie = count_race_dialogs_by_movie('The Fellowship of the Ring')
    # print(race_dialogs_by_movie)
    # draw_histogram_by_dictionary(race_dialogs_by_movie, 'amount of dialogs by race in part 1', 'value', 'race')
    #
    # race_dialogs_by_movie = count_race_dialogs_by_movie('The Two Towers')
    # # print(race_dialogs_by_movie)
    # draw_histogram_by_dictionary(race_dialogs_by_movie, 'amount of dialogs by race in part 2', 'value', 'race')
    #
    # race_dialogs_by_movie = count_race_dialogs_by_movie('The Return of the King')
    # print(race_dialogs_by_movie)
    # draw_histogram_by_dictionary(race_dialogs_by_movie, 'amount of dialogs by race in part 3', 'value', 'race')

    # def get_all_words():
    #     all_words = {}
    #     dialogs = list(vectorize_dialogs().columns)
    #     for dialogue in dialogs:
    #         words = pd.Series(dialogue.split(' ')).value_counts()
    #         for word in words.index:
    #             if word.index in all_words:
    #                 all_words[word.index] += words[word]
    #             else: all_words[word.index] = words[word]
    #
    #     print('Unique words count:', len(all_words))

    dialogs, labels = get_dialogs('race')

    all_words = {}
    for dialog in dialogs:
        if not isinstance(dialog, str):
            continue
        words = pd.Series(dialog.split(" ")).value_counts()
        for word in words.index:
            if word.index in all_words:
                all_words[word] += words[word]
            else:
                all_words[word] = words[word]

    print(f'Unique words count: {len(all_words)}')

    embedding_dimension = 128
    vocabulary_size = len(all_words)
    sequence_length = 64
    vect_layer = TextVectorization(max_tokens=vocabulary_size, output_mode="int",
                                   output_sequence_length=sequence_length)
    vect_layer.adapt(dialogs)

    # input_layer = Input(shape=(1,), dtype=tf_string)
    # vector = vect_layer(input_layer)
    # embedding = Embedding(vocabulary_size, embedding_dimension)(vector)
    # x = Bidirectional(LSTM(64, return_sequences=True))(embedding)
    # x = Dropout(0.5)(x)
    # x = Bidirectional(LSTM(32))(x)
    # x = Dropout(0.5)(x)
    # x = Dense(16, "relu")(x)
    # x = Dropout(0.5)(x)
    # output_layer = Dense(12, "softmax")(x)

    input_layer = Input(shape=(1,), dtype=tf_string)
    vector = vect_layer(input_layer)
    embedding = Embedding(vocabulary_size, embedding_dimension)(vector)
    x = LSTM(64, return_sequences=True)(embedding)
    x = Dropout(0.5)(x)
    x = LSTM(32)(x)
    x = Dropout(0.5)(x)
    x = Dense(16, "relu")(x)
    x = Dropout(0.5)(x)
    output_layer = Dense(12, "softmax")(x)

    model = Model(input_layer, output_layer)
    model.summary()
    model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])

    early_stopping = EarlyStopping(monitor="val_loss", min_delta=0, patience=30, restore_best_weights=30)
    batch_size = 32
    epochs = 30

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels) # pandas.core.series.Series

    X_train, X_test, y_train, y_test = train_test_split(dialogs, labels, test_size=0.2, random_state=11)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.20, random_state=1)

    y_train_vect = to_categorical(y_train)
    y_valid_vect = to_categorical(y_valid)

    X_train = np.asarray(X_train)
    X_valid = np.asarray(X_valid)

    history_bdirect = model.fit(X_train, y_train_vect, validation_data=(X_valid, y_valid_vect),
                                callbacks=[early_stopping], epochs=epochs, batch_size=batch_size)

    class_report(y_test, model.predict(X_test))
