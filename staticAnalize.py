import pandas as pd
import enums
import matplotlib.pyplot as plt
import numpy as np
import random
import string

LOTR_DATASETS = 'lotr_characters'
CLEAR_LOTR_DATASETS = 'cleaned_' + LOTR_DATASETS


def count_by_params(csv_file, columns):
    param = columns[-1]
    dictionary = {}
    csv_reader = pd.read_csv("./" + CLEAR_LOTR_DATASETS + '/' + csv_file, usecols=columns)
    for _, value in csv_reader.iterrows():
        dictionary[value[param]] = dictionary[value[param]] + 1 if value[param] in dictionary else 1

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

    return {k: v for k, v in sorted(dictionary.items(), key=lambda item: item[1], reverse=True)}


def count_dialogs_by_race():
    dictionary_dialogs = count_heroes_by_dialogs()
    dictionary_dialogs_lower = {}
    for key, value in dictionary_dialogs.items():
        dictionary_dialogs_lower[key.lower()] = value

    dictionary_chars = {}
    csv_reader = pd.read_csv("./" + CLEAR_LOTR_DATASETS + '/' + 'lotr_characters.csv', usecols=['name', 'race'])
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


def count_race_dialogs_by_movie(selected_movie):
    dictionary_dialogs = {}
    csv_reader = pd.read_csv("./" + CLEAR_LOTR_DATASETS + '/' + 'lotr_scripts.csv', usecols=['char', 'movie'])
    for _, value in csv_reader.iterrows():
        if value.movie not in dictionary_dialogs:
            dictionary_dialogs[value.movie] = {}
            dictionary_dialogs[value.movie][value.char] = 1
        elif value.char not in dictionary_dialogs[value.movie]:
            dictionary_dialogs[value.movie][value.char] = 1
        else:
            dictionary_dialogs[value.movie][value.char] += 1

    dictionary_chars = {}
    csv_reader = pd.read_csv("./" + CLEAR_LOTR_DATASETS + '/' + 'lotr_characters.csv', usecols=['name', 'race'])
    for _, value in csv_reader.iterrows():
        if not isinstance(value.race, str):
            continue
        if value.race not in dictionary_chars:
            dictionary_chars[value.race] = [value[0].lower()]
        else:
            dictionary_chars[value.race].append(value[0].lower())

    dictionary_race = {
        'The Return of the King': {},
        'The Two Towers': {},
        'The Fellowship of the Ring': {}
    }

    for _, movie in dictionary_race.items():
        for race, _ in dictionary_chars.items():
            movie[race] = 0

    for movie, dialogs in dictionary_dialogs.items():
        for character, character_dialogs in dialogs.items():
            for race, race_characters in dictionary_chars.items():
                if character.lower() in race_characters:
                    dictionary_race[movie.strip()][race] += character_dialogs

    return {k: v for k, v in sorted(dictionary_race[selected_movie].items(), key=lambda item: item[1], reverse=True)}


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


def draw_pie_chart(dictionary):
    get_n_pairs = {k: dictionary[k] for k in list(dictionary)}
    labels = list(get_n_pairs.keys())
    sizes = list(get_n_pairs.values())

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)

    plt.show()
