#!/usr/bin/python3

import os
import re

from staticAnalize import *
from vectoralizeDialog import get_dialogs_per_char, get_most_popular_phrase_by_char, get_dialog_sentiment

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
    csv_reader.to_csv("./" + CLEAR_LOTR_DATASETS + '/' + 'lotr_characters.csv')

    csv_reader = pd.read_csv("./" + LOTR_DATASETS + '/' + 'lotr_scripts.csv')
    del csv_reader['Unnamed: 0']
    for _, val in csv_reader.iterrows():
        val["char"] = val["char"].lower()
        val["char"] = enums.NAME_DICTIONARY[val["char"]] if val["char"] in enums.NAME_DICTIONARY else val["char"]
        val["dialog"] = " ".join(val["dialog"].split()) if isinstance(val["dialog"], str) else " "
        val["dialog"] = re.sub(r'(?<=[.,])(?=[^\s])', r' ', val["dialog"])

    csv_reader.to_csv("./" + CLEAR_LOTR_DATASETS + '/' + 'lotr_scripts.csv')


if __name__ == "__main__":
    get_kaggle_data()
    clear_data()

    print(count_heroes_by_dialogs())
    print(get_dialog_sentiment())
    print(get_most_popular_phrase_by_char())

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
    # print(race_dialogs_by_movie)
    # draw_histogram_by_dictionary(race_dialogs_by_movie, 'amount of dialogs by race in part 2', 'value', 'race')
    #
    # race_dialogs_by_movie = count_race_dialogs_by_movie('The Return of the King')
    # print(race_dialogs_by_movie)
    # draw_histogram_by_dictionary(race_dialogs_by_movie, 'amount of dialogs by race in part 3', 'value', 'race')
