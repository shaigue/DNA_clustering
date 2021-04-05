import math

def true_positive(true_clustering: list[set[int]], estimated_clustering: list[set[int]]) -> int:
    tp = 0
    for i in range(len(true_clustering)):
        for j in range(len(estimated_clustering)):
            x = len(true_clustering[i].intersection(estimated_clustering[j]))
            if x >= 2:
                tp = tp + x*(x-1)/2
    return tp

def false_positive(true_clustering: list[set[int]], estimated_clustering: list[set[int]]) -> int:
    fp = 0
    for i in range(len(true_clustering)):
        l_t = len(true_clustering[i])
        fp = fp + l_t*(l_t-1)/2
    return fp - true_positive(true_clustering, estimated_clustering)

def false_negative(true_clustering: list[set[int]], estimated_clustering: list[set[int]]) -> int:
    fn = 0
    for i in range(len(estimated_clustering)):
        l_t = len(estimated_clustering[i])
        fn = fn + l_t*(l_t-1)/2
    return fn - true_positive(true_clustering, estimated_clustering)

def fmi(true_clustering: list[set[int]], estimated_clustering: list[set[int]]) -> float:
    tp = true_positive(true_clustering, estimated_clustering)
    fp = false_positive(true_clustering, estimated_clustering)
    fn = false_negative(true_clustering, estimated_clustering)
    return tp / math.sqrt((tp + fp) * (tp + fn))