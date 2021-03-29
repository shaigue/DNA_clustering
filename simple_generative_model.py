import random

from dna_data_structure import DNASample, DNASampleSet
from complex_generative_model import sample_orig_strands, rand_symbol


def p_noisy_strand(original_strand: list[int], error_p: float) -> list[int]:
    error_strand = []
    for symbol in original_strand:
        r = random.random()
        # substitution
        if r <= error_p / 3:
            error_strand.append(rand_symbol())
        # deletion
        elif r <= 2 * error_p / 3:
            pass
        # insertion after
        elif r <= error_p:
            error_strand.append(symbol)
            error_strand.append(rand_symbol())
        # keep
        else:
            error_strand.append(symbol)
    return error_strand


def generate_dna_sample_set(n_clusters: int, cluster_size: int, error_p: float, strand_length: int) -> DNASampleSet:
    original_strands = sample_orig_strands(n_clusters, strand_length)
    error_strands = []
    for i in range(n_clusters):
        for j in range(cluster_size):
            error_strand = p_noisy_strand(original_strands[i], error_p)
            sample = DNASample(error_strand, i)
            error_strands.append(sample)

    return DNASampleSet(original_strands, error_strands)
