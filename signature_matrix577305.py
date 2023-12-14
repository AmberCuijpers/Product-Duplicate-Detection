import math
import re
import numpy as np
import pandas as pd


def model_words_title(input_title):
    regex = re.compile(
        r'(?:^|(?<=[ \[\(]))([a-zA-Z0-9]*(?:(?:[0-9]+[^0-9\., ()]+)|(?:[^0-9\., ()]+[0-9]+)|(?:([0-9]+\.[0-9]+)[^0-9\., ()]+))[a-zA-Z0-9]*)(?:$|(?=[ \)\]]))')
    model_word = [x for sublist in regex.findall(input_title) for x in sublist if x != ""]
    return model_word


# Brands consist of alphanumeric characters, so we want to keep this
def model_words_brand(input_brand):
    regex = re.compile(r'\b\w+\b')
    model_word = regex.findall(input_brand)
    return model_word


# Resolution consists of 3 or 4 digits and p
def model_words_resolution(input_resolution):
    regex = re.compile(r'\b(?:\w+|\d{3,4}p)\b')
    model_word = regex.findall(input_resolution)
    return model_word


def shingles(input_shingle, k):
    obtained_shingles = [input_shingle[i:i+k] for i in range(len(input_shingle) - k + 1)]
    return obtained_shingles


def obtaining_binary_vector(dataframe):
    number_rows = len(dataframe)
    model_words = []
    distinct_words = []

    for i in range(number_rows):
        model_words.append(model_words_title(dataframe['title'][i]))
        model_words.append(model_words_brand(dataframe['brand'][i]))
    # Want to have no duplicates of model words to make code more efficient
    for i in range(len(model_words)):
        words = shingles(model_words[i], 1)
        unique_words = [word for word in words if word not in distinct_words]
        distinct_words.extend(unique_words)

    binary_vectors = np.zeros((len(distinct_words), number_rows))
    for p in range(number_rows):
        # Iterate over all model words
        for mw in range(len(distinct_words)):
            # Check if any of the model words are in the title or brand strings
            if any(word in dataframe['title'][p] or word in dataframe['brand'][p] for word in distinct_words[mw]):
                binary_vectors[mw][p] = 1
    return binary_vectors


# Input also includes the fraction of rows that are used in the reduced set
def minhashing(fraction_rows, binary_vectors):
    amount_of_products = binary_vectors.shape[1]
    amount_of_model_words = len(binary_vectors)
    # Only a fraction of the rows is used, so compute how many
    number_min_hashes = math.floor(fraction_rows * amount_of_model_words)
    signature_matrix = np.zeros((number_min_hashes, amount_of_products))
    binary_vectors_copy = pd.DataFrame(binary_vectors, columns=np.arange(amount_of_products))
    for min_hash in range(number_min_hashes):
        binary_vectors_rearranged = binary_vectors_copy.sample(frac=1)
        for product in range(amount_of_products):
            # Algorithm of van Dam et al. (2016)
            column = binary_vectors_rearranged.iloc[:, product]
            x = np.where(column.values == 1)[0][0]
            signature_matrix[min_hash, product] = x
    return signature_matrix