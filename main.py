#!/usr/bin/python3

import os
import re

from staticAnalize import *
from vectoralizeDialog import *
from kmeans import *

os.environ['KAGGLE_USERNAME'] = "michalmichael"
os.environ['KAGGLE_KEY'] = "212e4a92b4a5f6143a6a3fc26c2375bd"

import kaggle
import pandas as pd
import enums

LOTR_DATASETS = 'lotr_characters'
CLEAR_LOTR_DATASETS = 'cleaned_' + LOTR_DATASETS


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

    # kmeans_results = run_KMeans(8, data)
    # kmeans = kmeans_results.get(3)

    db_scan = run_DBSan(data)
    final_df_array = data.to_numpy()
    n_feats = 10
    dfs = get_top_features_cluster(final_df_array, db_scan, n_feats, vectorizer)
    plotWords(dfs, 13)

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
