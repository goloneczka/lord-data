import os
import random
import string

os.environ['KAGGLE_USERNAME'] = "michalmichael"
os.environ['KAGGLE_KEY'] = "212e4a92b4a5f6143a6a3fc26c2375bd"

import kaggle
import pandas as pd
import enums
import matplotlib.pyplot as plt
import numpy as np


LOTR_DATASETS = 'lotr_characters.csv'


def get_kaggle_data():
    try:
        os.mkdir(LOTR_DATASETS)
    except OSError:
        print("Structured dir created - skip structuring data")
        return

    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('paultimothymooney/lord-of-the-rings-data', path=LOTR_DATASETS,
                                      unzip=True)


def count_by_params(csv_file, columns):
    param = columns[-1]
    dictionary = {}
    csv_reader = pd.read_csv("./" + LOTR_DATASETS + '/' + csv_file, usecols=columns)
    for _, value in csv_reader.iterrows():
        alias = enums.NAME_DICTIONARY[value[param]] if value[param] in enums.NAME_DICTIONARY else value[param]
        alias = 'UNKNOWN' if not isinstance(alias, str) else alias
        dictionary[alias] = dictionary[alias] + 1 if alias in dictionary else 1

    return {k: v for k, v in sorted(dictionary.items(), key=lambda item: item[1], reverse=True)}


def count_races_by_heroes():
    return count_by_params('lotr_characters.csv', ['name', 'race'])


def count_heroes_by_dialogs():
    return count_by_params('lotr_scripts.csv', ['dialog', 'char'])

def count_dialogs_by_movie():
    return count_by_params('lotr_scripts.csv', ['dialog', 'movie'])

def count_heroes_by_gender():
    return count_by_params('lotr_characters.csv', ['name', 'gender'])


def average_dialogs_length_by_hero():
    # top 15 most popular hereos
    # most_popular_heroes = list(count_heroes_by_dialogs().keys())[:15]
    columns = ['dialog', 'char']
    by_param = columns[0]
    to_find_param = columns[1]
    dictionary = {}
    csv_reader = pd.read_csv("./" + LOTR_DATASETS + '/' + 'lotr_scripts.csv', usecols=columns)
    for _, value in csv_reader.iterrows():
        try:
            clear_dialog = " ".join(value[by_param].split())
            lenght_of_dialog = len(clear_dialog.split(sep=" "))
        except AttributeError:
            continue
        alias = enums.NAME_DICTIONARY[value[to_find_param]] if value[to_find_param] in enums.NAME_DICTIONARY else value[
            to_find_param]
        dictionary[alias] = dictionary[alias] + lenght_of_dialog if alias in dictionary else lenght_of_dialog

    dictionary_with_dialogs = count_heroes_by_dialogs()
    for key, value in dictionary.items():
        dictionary[key] = round(value / dictionary_with_dialogs[key], 2)

    # removing less important hereoes
    # for key in dictionary.copy().keys():
    #     if key not in most_popular_heroes:
    #         del dictionary[key]

    return {k: v for k, v in sorted(dictionary.items(), key=lambda item: item[1], reverse=True)}


def count_dialogs_by_race():

    dictionary_dialogs = count_heroes_by_dialogs()
    dictionary_dialogs_lower = {}
    for key, value in dictionary_dialogs.items():
        dictionary_dialogs_lower[key.lower()] = value

    dictionary_chars = {}
    csv_reader = pd.read_csv("./" + LOTR_DATASETS + '/' + 'lotr_characters.csv', usecols=['name', 'race'])
    for _, value in csv_reader.iterrows():
        race = 'UNKNOWN' if not isinstance(value.race, str) else value.race
        if race not in dictionary_chars:
            dictionary_chars[race] = [value[0].lower()]
        else:
            dictionary_chars[race].append(value[0].lower())

    dictionary_race = {}
    for key, nested_dictionary in dictionary_chars.items():
        dictionary_race[key] = 0
        for hero in nested_dictionary:
            alias = enums.NAME_DICTIONARY[hero] if hero in enums.NAME_DICTIONARY else hero
            if alias in dictionary_dialogs_lower:
                dictionary_race[key] += dictionary_dialogs_lower[alias]


    return {k: v for k, v in sorted(dictionary_race.items(), key=lambda item: item[1], reverse=True)}


def count_gender_dialogs_by_move():
    dictionary_dialogs = {}
    csv_reader = pd.read_csv("./" + LOTR_DATASETS + '/' + 'lotr_scripts.csv', usecols=['char', 'movie'])
    for _, value in csv_reader.iterrows():
        if value.movie not in dictionary_dialogs:
            dictionary_dialogs[value.movie] = {}
            dictionary_dialogs[value.movie][value.char] = 1
        elif value.char not in dictionary_dialogs[value.movie]:
            dictionary_dialogs[value.movie][value.char] = 1
        else:
            dictionary_dialogs[value.movie][value.char] += 1

    dictionary_chars = {}
    csv_reader = pd.read_csv("./" + LOTR_DATASETS + '/' + 'lotr_characters.csv', usecols=['name', 'gender'])
    for _, value in csv_reader.iterrows():
        if value.gender not in dictionary_chars:
            dictionary_chars[value.gender] = [value[1].lower()]
        else:
            dictionary_chars[value.gender].append(value[1].lower())

    dictionary_gender = {}
    for key, _ in dictionary_dialogs.items():
        dictionary_gender[key] = {'Female': 0, 'Male': 0}
    for key, nested_dictionary in dictionary_dialogs.items():
        for nested_key, nested_value in nested_dictionary.items():
            alias = enums.NAME_DICTIONARY[nested_key] if nested_key in enums.NAME_DICTIONARY else nested_key
            if alias.lower() in dictionary_chars['Female']:
                dictionary_gender[key]['Female'] += nested_value
            elif alias.lower() in dictionary_chars['Male']:
                dictionary_gender[key]['Male'] += nested_value

    return dictionary_gender


def draw_histogram_by_dictionary(dictionary, title, ylabel, xlabel):

    count = 6 if 'UNKNOWN' in dictionary else 5
    get_n_pairs = {k: dictionary[k] for k in list(dictionary)[:count]}
    names = list(get_n_pairs.keys())
    values = list(get_n_pairs.values())

    plt.bar(range(len(get_n_pairs)), values, tick_label=names, width=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis='y', alpha=0.75)
    plt.show()

    plt_name = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
    plt.savefig(plt_name + '.png')


def draw_histogram_by_dictionary_with_dictionary(dictionary, title, ylabel, xlabel):
    names = []
    females = []
    males = []
    for key, nested_dictionary in dictionary.items():
        names.append(key)
        females.append(nested_dictionary['Female'])
        males.append(nested_dictionary['Male'])

    x_axis = np.arange(len(names))
    plt.bar(x_axis - 0.2, females, tick_label=names, width=0.4)
    plt.bar(x_axis + 0.2, males, tick_label=names, width=0.4)
    plt.yscale('log')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis='y', alpha=0.75)
    plt.show()

    plt_name = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
    plt.savefig(plt_name + '.png')


def draw_pie_chart(dictionary):
    get_n_pairs = {k: dictionary[k] for k in list(dictionary)}
    labels = list(get_n_pairs.keys())
    sizes = list(get_n_pairs.values())

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)

    plt.show()


if __name__ == "__main__":
    get_kaggle_data()
    print(count_dialogs_by_race())
    draw_histogram_by_dictionary(count_dialogs_by_race(), 'amount of dialogs by race', 'value', 'race')
    print(count_races_by_heroes())
    print(count_dialogs_by_movie())
    draw_histogram_by_dictionary(count_races_by_heroes(), 'amount of heroes in races', 'value', 'race')
    draw_histogram_by_dictionary(count_dialogs_by_movie(), 'amount of dialogs per movie', 'value', 'movie')
    draw_pie_chart(count_heroes_by_gender())
    print(count_heroes_by_dialogs())
    draw_histogram_by_dictionary(count_heroes_by_dialogs(), 'amount of dialogs by hero', 'value', 'hero')
    print(average_dialogs_length_by_hero())
    draw_histogram_by_dictionary(average_dialogs_length_by_hero(), 'average length of sentence by hero', 'value', 'hero')
    print(count_gender_dialogs_by_move())
    draw_histogram_by_dictionary_with_dictionary(count_gender_dialogs_by_move(), 'sentences by gender in movies', 'value', 'movie')
