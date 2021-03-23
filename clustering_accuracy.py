"""This is a module to calculate the clustering accuracy from the paper"""
import numpy as np


def convert_cluster_labels_to_partition(labels: np.ndarray):
    n_clusters = max(labels) + 1
    partition = [set() for _ in range(n_clusters)]
    for i, label in enumerate(labels):
        partition[label].add(i)
    return partition


def check_valid_partition(partition: list[set[int]]):
    # check pairwise disjunction
    n = len(partition)
    for i in range(n):
        for j in range(n):
            if i != j:
                intersect = partition[i].intersection(partition[j])
                if len(intersect) != 0:
                    return False
    # check full set
    union = set()
    for subset in partition:
        union.update(subset)
    max_value = max(union)
    return union == set(range(max_value + 1))


def clustering_accuracy(true_clustering: list[set[int]], estimated_clustering: list[set[int]],
                        min_part_coef: float = 1) -> float:
    """We assume that the input is a partition of the set {0,...,n}

    :param true_clustering:
    :param estimated_clustering:
    :param min_part_coef:
    :return: the clustering accuracy
    """
    assert check_valid_partition(estimated_clustering)
    assert check_valid_partition(true_clustering)
    assert 0.5 < min_part_coef <= 1

    # because min_part_coef > 0.5, and the subset pairwise distinct,
    # then each true cluster can match at most one estimated cluster
    matching_clusters = 0
    for true_cluster in true_clustering:
        for estimated_cluster in estimated_clustering:
            if estimated_cluster.issubset(true_cluster) and \
                    len(estimated_cluster.intersection(true_cluster)) > min_part_coef * len(true_cluster):
                matching_clusters += 1
    return matching_clusters / len(true_clustering)

