import json
import random
import numpy as np
import matplotlib.pyplot as plt
from data_cleaning577305 import create_dataframe, completing_dataframe, cleaning_data, duplicates_counter
from signature_matrix577305 import obtaining_binary_vector, minhashing
from LSH577305 import locality_sensitive_hashing
from MSM577305 import create_dissimilarity_matrix, clustering, best_threshold_clustering
from evaluations577305 import performance_measure, performance_measure_lsh
from bootstrapCode577305 import get_training_test_data

path = '/Volumes/data/Documenten/Econometrie/Master/Advanced Computer Science/TVs-all-merged.json'
with open(path, 'r') as file:
    # Load the JSON data
    data = json.load(file)
dataframe = create_dataframe(data)
new_dataframe = completing_dataframe(dataframe)
cleaned_dataframe = cleaning_data(new_dataframe)

thresholds = np.arange(0, 1.05, 0.05)

precision_training_values = []
recall_training_values = []
F1_training_values = []
pair_quality_training_values = []
pair_completeness_training_values = []
F1_star_training_values = []
for threshold in thresholds:
    print("Threshold:", threshold)
    number_of_bootstraps = 5
    total_number_of_comparisons_training = []
    total_comparisons_fraction_training = []
    total_precision_training = []
    total_recall_training = []
    total_F1_training = []

    total_number_of_comparisons_test = []
    total_comparisons_fraction_test = []
    total_precision_test = []
    total_recall_test = []
    total_F1_test = []

    total_number_of_comparisons_lsh_training = []
    total_comparisons_fraction_lsh_training = []
    total_pair_quality_training = []
    total_pair_completeness_training = []
    total_F1_star_training = []

    total_number_of_comparisons_lsh_test = []
    total_comparisons_fraction_lsh_test = []
    total_pair_quality_test = []
    total_pair_completeness_test = []
    total_F1_star_test = []

    for bootstrap in range(1, number_of_bootstraps + 1):
        random.seed(bootstrap)
        print("Bootstrap:", bootstrap)
        number_of_products = len(cleaned_dataframe)
        training_data, test_data = get_training_test_data(cleaned_dataframe, number_of_products)

        print("Training of bootstrap:", bootstrap)
        binary_vector_training = obtaining_binary_vector(training_data)
        signature_matrix_training = minhashing(0.5, binary_vector_training)
        lsh_training = locality_sensitive_hashing(signature_matrix_training, 0.5)
        dissimilarity_matrix_training = create_dissimilarity_matrix(lsh_training, training_data, 4)

        duplicate_matrix_training, real_duplicates_training = duplicates_counter(training_data)
        best_threshold_training = best_threshold_clustering(dissimilarity_matrix_training, training_data, real_duplicates_training)
        clustering_duplicates_training = clustering(dissimilarity_matrix_training, best_threshold_training)


        number_of_comparisons_training, comparisons_fraction_training, precision_training, recall_training, \
            F1_training = performance_measure(clustering_duplicates_training, real_duplicates_training, training_data)
        total_number_of_comparisons_training.append(number_of_comparisons_training)
        total_comparisons_fraction_training.append(comparisons_fraction_training)
        total_precision_training.append(precision_training)
        total_recall_training.append(recall_training)
        total_F1_training.append(F1_training)

        number_of_comparisons_lsh_training, comparisons_fraction_lsh_training, pair_quality_training, \
            pair_completeness_training, F1_star_training = performance_measure_lsh(lsh_training, training_data,
                                                                                       duplicate_matrix_training)
        total_number_of_comparisons_lsh_training.append(number_of_comparisons_lsh_training)
        total_comparisons_fraction_lsh_training.append(comparisons_fraction_lsh_training)
        total_pair_quality_training.append(pair_quality_training)
        total_pair_completeness_training.append(pair_completeness_training)
        total_F1_star_training.append(F1_star_training)

        print("Test of bootstrap:", bootstrap)
        binary_vector_test = obtaining_binary_vector(test_data)
        signature_matrix_test = minhashing(0.5, binary_vector_test)
        lsh_test = locality_sensitive_hashing(signature_matrix_test, threshold)
        dissimilarity_matrix_test = create_dissimilarity_matrix(lsh_test, test_data, 4)

        duplicate_matrix_test, real_duplicates_test = duplicates_counter(test_data)
        best_threshold_test = best_threshold_clustering(dissimilarity_matrix_test, test_data, real_duplicates_test)
        clustering_duplicates_test = clustering(dissimilarity_matrix_test, best_threshold_test)

        number_of_comparisons_test, comparisons_fraction_test, precision_test, recall_test, \
            F1_test = performance_measure(clustering_duplicates_test, real_duplicates_test, test_data)
        total_number_of_comparisons_test.append(number_of_comparisons_test)
        total_comparisons_fraction_test.append(comparisons_fraction_test)
        total_precision_test.append(precision_test)
        total_recall_test.append(recall_test)
        total_F1_test.append(F1_test)

        number_of_comparisons_lsh_test, comparisons_fraction_lsh_test, pair_quality_test, \
            pair_completeness_test, F1_star_test = performance_measure_lsh(lsh_test, test_data,
                                                                                       duplicate_matrix_test)
        total_number_of_comparisons_lsh_test.append(number_of_comparisons_lsh_test)
        total_comparisons_fraction_lsh_test.append(comparisons_fraction_lsh_test)
        total_pair_quality_test.append(pair_quality_test)
        total_pair_completeness_test.append(pair_completeness_test)
        total_F1_star_test.append(F1_star_test)
    average_precision_training = np.mean(total_precision_training)
    precision_training_values.append(average_precision_training)
    average_recall_test = np.mean(total_recall_training)
    recall_training_values.append(average_recall_test)
    average_F1_test = np.mean(total_F1_training)
    F1_training_values.append(average_F1_test)
    average_pair_quality = np.mean(total_pair_quality_training)
    pair_quality_training_values.append(average_pair_quality)
    average_pair_completeness = np.mean(total_pair_completeness_training)
    pair_completeness_training_values.append(average_pair_completeness)
    average_F1_star = np.mean(total_F1_star_training)
    F1_star_training_values.append(average_F1_star)

plt.plot(thresholds, precision_training_values, marker='o')
plt.xlabel('Threshold')
plt.ylabel('Precision')
plt.show()

plt.plot(thresholds, recall_training_values, marker='o')
plt.xlabel('Threshold')
plt.ylabel('Recall')
plt.show()

plt.plot(thresholds, F1_training_values, marker='o')
plt.xlabel('Threshold')
plt.ylabel('F1')
plt.show()

plt.plot(thresholds, pair_quality_training_values, marker='o')
plt.xlabel('Threshold')
plt.ylabel('Pair quality')
plt.show()

plt.plot(thresholds, pair_completeness_training_values, marker='o')
plt.xlabel('Threshold')
plt.ylabel('Pair completeness')
plt.show()

plt.plot(thresholds, F1_star_training_values, marker='o')
plt.xlabel('Threshold')
plt.ylabel('F1_star')
plt.show()
print(1)
