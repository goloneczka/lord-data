import os

os.environ['KAGGLE_USERNAME'] = "michalmichael"
os.environ['KAGGLE_KEY'] = "XXXX"  # TODO secure this key

import kaggle
import pandas as pd
import enums

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
        dictionary[alias] = dictionary[alias] + 1 if alias in dictionary else 1

    return {k: v for k, v in sorted(dictionary.items(), key=lambda item: item[1], reverse=True)}


def count_races_by_heroes():
    return count_by_params('lotr_characters.csv', ['name', 'race'])


def count_heroes_by_dialogs():
    return count_by_params('lotr_scripts.csv', ['dialog', 'char'])


def average_dialogs_length_by_hero(csv_file, columns):
    by_param = columns[0]
    to_find_param = columns[1]
    dictionary = {}
    csv_reader = pd.read_csv("./" + LOTR_DATASETS + '/' + csv_file, usecols=columns)
    for _, value in csv_reader.iterrows():
        try:
            clear_dialog = " ".join(value[by_param].split())
            lenght_of_dialog = len(clear_dialog.split(sep=" "))
        except AttributeError:
            continue
        alias = enums.NAME_DICTIONARY[value[to_find_param]] if value[to_find_param] in enums.NAME_DICTIONARY else value[to_find_param]
        dictionary[alias] = dictionary[alias] + lenght_of_dialog if alias in dictionary else lenght_of_dialog

    dictionary_with_dialogs = count_heroes_by_dialogs()
    for key, value in dictionary.items():
        dictionary[key] = round(value / dictionary_with_dialogs[key], 2)

    return {k: v for k, v in sorted(dictionary.items(), key=lambda item: item[1], reverse=True)}

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

    print(dictionary_gender)


if __name__ == "__main__":
    get_kaggle_data()
    print(count_races_by_heroes())
    print(count_heroes_by_dialogs())
    print(average_dialogs_length_by_hero('lotr_scripts.csv', ['dialog', 'char']))
    count_gender_dialogs_by_move()
