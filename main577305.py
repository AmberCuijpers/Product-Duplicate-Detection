import json
import matplotlib.pyplot as plt
import numpy as np

from data_cleaning577305 import create_dataframe, completing_dataframe, cleaning_data, duplicates_counter
from signature_matrix577305 import obtaining_binary_vector, minhashing
from LSH577305 import locality_sensitive_hashing
from MSM577305 import create_dissimilarity_matrix, clustering, best_threshold_clustering
from evaluations577305 import performance_measure, performance_measure_lsh

path = '/Volumes/data/Documenten/Econometrie/Master/Advanced Computer Science/TVs-all-merged.json'
with open(path, 'r') as file:
    # Load the JSON data
    data = json.load(file)
dataframe = create_dataframe(data)

new_dataframe = completing_dataframe(dataframe).head(10)
cleaned_dataframe = cleaning_data(new_dataframe)
binary_vector = obtaining_binary_vector(cleaned_dataframe)
thresholds = np.arange(0, 1.05, 0.05)

precision_values = []
recall_values = []
F1_values = []
pair_quality_values = []
pair_completeness_values = []
F1_star_values = []
for threshold in thresholds:
    signature_matrix = minhashing(0.5, binary_vector)
    lsh = locality_sensitive_hashing(signature_matrix, 0.5)
    dissimilarity_matrix = create_dissimilarity_matrix(lsh, cleaned_dataframe, 4)

    duplicate_matrix, real_duplicates = duplicates_counter(new_dataframe)
    best_threshold = best_threshold_clustering(dissimilarity_matrix, cleaned_dataframe, real_duplicates)
    clustering_duplicates = clustering(dissimilarity_matrix, best_threshold)


    number_of_comparisons, comparisons_fraction, precision, recall, F1 = performance_measure(clustering_duplicates, real_duplicates, cleaned_dataframe)
    number_of_comparisons_lsh, comparisons_fraction_lsh, pair_quality, pair_completeness, f1_star = \
        performance_measure_lsh(lsh, cleaned_dataframe, duplicate_matrix)
    precision_values.append(precision)
    recall_values.append(recall)
    F1_values.append(F1)
    pair_quality_values.append(pair_quality)
    pair_completeness_values.append(pair_completeness)
    F1_star_values.append(f1_star)

plt.plot(thresholds, precision_values, marker='o')
plt.xlabel('Threshold')
plt.ylabel('Precision')
plt.show()

plt.plot(thresholds, recall_values, marker='o')
plt.xlabel('Threshold')
plt.ylabel('Recall')
plt.show()

plt.plot(thresholds, F1_values, marker='o')
plt.xlabel('Threshold')
plt.ylabel('F1')
plt.show()

plt.plot(thresholds, pair_quality_values, marker='o')
plt.xlabel('Threshold')
plt.ylabel('Pair quality')
plt.show()

plt.plot(thresholds, pair_completeness_values, marker='o')
plt.xlabel('Threshold')
plt.ylabel('Pair completeness')
plt.show()

plt.plot(thresholds, F1_star_values, marker='o')
plt.xlabel('Threshold')
plt.ylabel('F1_star')
plt.show()
print(1)