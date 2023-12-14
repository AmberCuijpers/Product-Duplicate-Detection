import numpy as np
from sklearn.cluster import AgglomerativeClustering
from evaluations577305 import performance_measure
from signature_matrix577305 import shingles


def create_dissimilarity_matrix(pairs_matrix, dataframe, k):
    amount_of_products = len(pairs_matrix)
    dissimilarity_matrix = np.zeros((amount_of_products, amount_of_products))
    for product_i in range(amount_of_products):
        for product_j in range(product_i + 1, amount_of_products):
            if pairs_matrix[product_i, product_j] != 0:
                # Only calculate the shingles and dissimilarity if it is a possible pair
                shingles1 = shingles(dataframe['title'][product_i], k)
                shingles2 = shingles(dataframe['title'][product_j], k)
                set_shingles1 = set(shingles1)
                set_shingles2 = set(shingles2)
                dissimilarity = jaccard_dissimilarity(set_shingles1, set_shingles2)
                dissimilarity_matrix[product_i, product_j] = dissimilarity
                dissimilarity_matrix[product_j, product_i] = dissimilarity
            if pairs_matrix[product_i, product_j] == 0:
                # Infinity is too large to calculate with during clustering thus therefore we choose a large number
                dissimilarity_matrix[product_i, product_j] = 1000
                dissimilarity_matrix[product_j, product_i] = 1000
            elif dataframe['shop'][product_i] == dataframe['shop'][product_j]:
                dissimilarity_matrix[product_i, product_j] = 1000
                dissimilarity_matrix[product_j, product_i] = 1000
            elif dataframe['brand'][product_i] != dataframe['brand'][product_j]:
                dissimilarity_matrix[product_i, product_j] = 1000
                dissimilarity_matrix[product_j, product_i] = 1000
            elif dataframe['resolution'][product_i] != dataframe['resolution'][product_j]:
                dissimilarity_matrix[product_i, product_j] = 1000
                dissimilarity_matrix[product_j, product_i] = 1000
    return dissimilarity_matrix


def jaccard_dissimilarity(A, B):
    similarity = len(A.intersection(B)) / len(A.union(B))
    dissimilarity = 1 - similarity
    return dissimilarity


def clustering(dissimilarity_matrix, threshold):
    linkage_clustering = AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="complete",
                                                 distance_threshold=threshold)
    clusters = linkage_clustering.fit_predict(dissimilarity_matrix)
    cluster_dict = {}
    pairs = set()

    for i, cluster_id in enumerate(clusters):
        if cluster_id not in cluster_dict:
            cluster_dict[cluster_id] = []
        cluster_dict[cluster_id].append(i)

    for cluster_indices in cluster_dict.values():
        if len(cluster_indices) > 1:
            for i in range(len(cluster_indices)):
                for j in range(i + 1, len(cluster_indices)):
                    if cluster_indices[i] < cluster_indices[j]:
                        pairs.add((cluster_indices[i], cluster_indices[j]))
    return pairs


def best_threshold_clustering(dissimilarity_matrix, dataframe, real_duplicates):
    best_F1 = 0
    best_threshold = 0
    # We do not want a threshold that is too low, so start at 60%
    thresholds = np.arange(0.6, 1.05, 0.05)
    for current_threshold in thresholds:
        # current_threshold = threshold_index / 20
        clustering_duplicates = clustering(dissimilarity_matrix, current_threshold)

        _, _, _, _, current_F1 = performance_measure(clustering_duplicates, real_duplicates, dataframe)

        if current_F1 > best_F1:
            best_F1 = current_F1
            best_threshold = current_threshold

    return best_threshold
