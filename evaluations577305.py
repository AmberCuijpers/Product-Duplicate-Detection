from math import comb
import numpy as np


def performance_measure(clustering_duplicates, real_duplicates, dataframe):
    amount_of_duplicates = 0
    for pair in clustering_duplicates:
        pair_product1 = pair[0]
        pair_product2 = pair[1]
        model_ID_product1 = dataframe['modelID'][pair_product1]
        model_ID_product2 = dataframe['modelID'][pair_product2]
        if model_ID_product1 == model_ID_product2:
            amount_of_duplicates += 1

    number_of_comparisons = len(clustering_duplicates)
    pair_quality = amount_of_duplicates / number_of_comparisons
    pair_completeness = amount_of_duplicates / real_duplicates

    F1 = (2 * pair_quality * pair_completeness) / (pair_quality + pair_completeness)

    n = len(dataframe)
    number_of_possible_comparisons = (n * (n - 1)) / 2
    comparisons_fraction = number_of_comparisons / number_of_possible_comparisons
    return number_of_comparisons, comparisons_fraction, pair_quality, pair_completeness, F1


def performance_measure_lsh(candidates, dataframe, duplicate_matrix):
    number_of_comparisons = np.sum(candidates)/2
    comparisons_fraction = number_of_comparisons / comb(len(dataframe), 2)
    correct = np.where(duplicate_matrix + candidates == 2, 1, 0)
    n_correct = np.sum(correct) / 2

    pair_quality = n_correct / number_of_comparisons
    pair_completeness = n_correct / (np.sum(duplicate_matrix) / 2)
    f1_star = 2 * pair_quality * pair_completeness / (pair_quality + pair_completeness)

    return number_of_comparisons, comparisons_fraction, pair_quality, pair_completeness, f1_star
