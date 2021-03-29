"""This is a module to calculate the clustering accuracy from the paper"""
import numpy as np


def convert_cluster_labels_to_partition(labels: np.ndarray) -> list[set[int]]:
    """Receives per sample cluster labels, and returns a partition - a list of sets that represents the different
    clusters.

    :param labels: an array with cluster labels of the i'th sample in the i'th entry
    :return a list of sets (a partition of all the labels) where each set is a cluster.
    """
    n_clusters = max(labels) + 1
    partition = [set() for _ in range(n_clusters)]
    for i, label in enumerate(labels):
        partition[label].add(i)
    return partition


def check_valid_partition(partition: list[set[int]]) -> bool:
    """Checks that a list of sets is indeed a partition, i.e. all the sets are mutually exclusive, and they cover all
    the samples from 0-n.

    :param partition: this is a list of the clusters
    :return True if it is a partition, otherwise False
    """
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
    """This is the clustering accuracy described in "clustering billions of DNA" paper.
    We assume that the input is a partition of the set {0,...,n}

    :param true_clustering: ground truth clustering of each sample
    :param estimated_clustering: estimated clustering partition
    :param min_part_coef: this is the lambda parameter described in the paper, represents what fraction of the original
        cluster has to be present in the estimated cluster so it will count as a good cluster.
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

