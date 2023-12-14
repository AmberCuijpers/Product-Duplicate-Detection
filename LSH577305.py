import math
import sys
import numpy as np


def locality_sensitive_hashing(signature_matrix, threshold):
    # Convert the values of the signature matrix to the closest integer value (because a lot are .000 which can be
    # rounded off
    signature_matrix = signature_matrix.astype(int)
    column_length = len(signature_matrix)
    row_length = len(signature_matrix[0])
    # The number of rows is the number of rows for each band. It must hold that length of columns signature matrix = r*b
    number_of_rows, number_of_bands = threshold_approximation(column_length, threshold)
    candidate_pairs_matrix = np.zeros((row_length, row_length))
    for current_band in range(number_of_bands):
        buckets = dict()
        first_row_current_band = number_of_rows * current_band
        last_row_current_band = number_of_rows * (current_band + 1)
        band_strings = ["".join(signature_matrix[first_row_current_band:last_row_current_band, column].astype(str))
                        for column in range(len(signature_matrix[0]))]
        band_hashes = [hash(string) % sys.maxsize for string in band_strings]

        # Add the item hashes to the corresponding bucket
        for current_item in range(len(band_hashes)):
            hash_value = band_hashes[current_item]
            if hash_value in buckets:

                # All items already in this bucket may be duplicates
                for candidate in buckets[hash_value]:
                    candidate_pairs_matrix[current_item, candidate] = 1
                    candidate_pairs_matrix[candidate, current_item] = 1
                buckets[hash_value].append(current_item)
            else:
                # If there is no item in the bucket, add current item
                buckets[hash_value] = [current_item]
    return candidate_pairs_matrix.astype(int)


def threshold_approximation(column_length, threshold):
    # Current number of bands and number of rows per band. These will be updated
    best_number_of_bands = 1
    best_number_of_rows = 1
    # The best approximation of the threshold until now
    best_approximation = 1
    for current_number_of_rows in range(1, math.isqrt(column_length) + 1):
        if column_length % current_number_of_rows == 0:
            # Only look at number of rows which evenly divide the column length, because n = r * b
            # Also, then b = n / r
            current_number_of_bands = column_length // current_number_of_rows
            current_approximation = (1 / current_number_of_bands) ** (1 / current_number_of_rows)
            if abs(current_approximation - threshold) < abs(best_approximation - threshold):
                best_approximation = current_approximation
                best_number_of_rows = current_number_of_rows
                best_number_of_bands = current_number_of_bands
    return best_number_of_rows, best_number_of_bands


