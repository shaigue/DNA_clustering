"""This is a file for the complex generative DNA storage channel"""
import math
import random
import time
from typing import List

from dna_data_structure import DNASample, DNASampleSet, C, G


def rand_symbol():
    """Generates a random symbol from 0, 1, 2, 3"""
    return random.randint(0, 3)


def rand_sub(symbol: int) -> int:
    """Chooses a random symbol out of 0, 1, 2, 3, except from the given symbol."""
    return random.choice(list({0, 1, 2, 3} - {symbol}))


def sample_orig_strands(n_samples: int, length: int):
    """Samples the strands with uniform distribution"""
    samples = []
    for i in range(n_samples):
        strand = []
        for j in range(length):
            strand.append(rand_symbol())
        samples.append(strand)
    return samples


def longest_run(strand: List[int]) -> int:
    """Calculates the length of the longest run (= same symbol) in the strand."""
    max_run = 0
    symbol = strand[0]
    curr_run = 0
    for s in strand:
        if s == symbol:
            curr_run += 1
            if curr_run > max_run:
                max_run = curr_run
        else:
            symbol = s
            curr_run = 1
    return max_run


def synthesis_term_var(strand: List[int]) -> float:
    """Calculates a factor that gives higher probability for strands with long runs to terminate"""
    cut_off = 2 + math.log2(len(strand)) / 2
    max_run = min(cut_off, longest_run(strand))
    return max_run / cut_off


def synthesis(orig_strands: List[List[int]], physical_redundancy: int, p_sub: float, p_del: float, p_ins: float,
              p_term_max: float) -> List[DNASample]:
    """Simulates the synthesis process.

    :param orig_strands: a list of the original strands
    :param physical_redundancy: number of copies produces for each original strand
    :param p_sub: probability for substitution
    :param p_del: probability for deletion
    :param p_ins: probability for insertion
    :param p_term_max: maximum probability for termination (per strand variance)
    """
    samples = []
    for i in range(len(orig_strands)):
        for j in range(physical_redundancy):
            orig_strand = orig_strands[i]
            p_term = p_term_max * synthesis_term_var(orig_strand)
            # early termination
            if random.random() >= p_term:
                syn_strand = []
                for symbol in orig_strand:
                    r = random.random()
                    # substitution
                    if r < p_sub:
                        syn_strand.append(rand_sub(symbol))
                    # deletion
                    elif r < p_sub + p_del:
                        pass
                    # insertion
                    elif r < p_sub + p_del + p_ins:
                        syn_strand.append(rand_symbol())
                        syn_strand.append(symbol)
                    else:
                        syn_strand.append(symbol)

                samples.append(DNASample(syn_strand, i))

    return samples


def storage(samples: List[DNASample], p_decay: List[float]) -> List[DNASample]:
    """Simulates decay of nuclei in storage, each nuclei has some probability to go to -1, and will not be duplicated in
    PCR, and will be randomly sequenced symbol.

    :param samples: the DNA pool simulated strands
    :param p_decay: a list with 4 entries with decay rate for each symbol
    """
    for sample in samples:
        for i, symbol in enumerate(sample.strand):
            r = random.random()
            if r < p_decay[symbol]:
                sample.strand[i] = -1
    return samples


def pcr_duplicate_prob(strand: List[int]) -> float:
    """Gives higher chance for duplication for low CG contents, and low for high CG contents"""
    strand_length = len(strand)
    cg_count = sum(1 for symbol in strand if symbol in [C, G])
    cg_rate = cg_count / strand_length
    return 1 - 0.25 * cg_rate


def pcr(samples: List[DNASample]) -> List[DNASample]:
    """Simulates a PCR cycle where some of the strands will be duplicated, and duplication rates are strand specific"""
    n_samples = len(samples)
    for i in range(n_samples):
        sample = samples[i]
        p = pcr_duplicate_prob(sample.strand)
        r = random.random()
        if r < p:
            samples.append(DNASample(sample.strand.copy(), sample.orig_idx))
    return samples


def sampling(samples: List[DNASample], n_samples: int) -> List[DNASample]:
    """Sample a random subset for sequencing"""
    return random.sample(samples, n_samples)


def sequencing(samples: List[DNASample], p_sub: float, ends_factor: float) -> List[DNASample]:
    """Simulates the sequencing procedure.
    some substitutions may occur, and they are more likely at the ends of the strands.

    :param samples: DNA samples
    :param p_sub: probability for substitution
    :param ends_factor: multiplicative factor to increase the error rates in the start & end of the strand
    """
    def position_sub_p(i: int, strand_length: int) -> float:
        if i < 10 or i > strand_length - 10:
            return p_sub * ends_factor
        return p_sub

    for sample in samples:
        for i, symbol in enumerate(sample.strand):
            if symbol == -1:
                sample.strand[i] = rand_symbol()
            else:
                p = position_sub_p(i, len(sample.strand))
                r = random.random()
                if r < p:
                    sample.strand[i] = rand_sub(symbol)

    return samples


def create_dna_samples(
        n_orig_strands: int = 100,
        length: int = 150,
        physical_redundancy: int = 1000,
        p_sub_syn: float = 0.02,
        p_del_syn: float = 0.01,
        p_ins_syn: float = 0.005,
        p_term_max_syn: float = 0.1,
        p_decay: List[float] = None,
        pcr_rounds: int = 5,
        n_final_samples: int = 1_000_000,
        p_sub_seq: float = 0.005,
        ends_factor_seq: float = 4) -> DNASampleSet:
    """A function that simulates the process of DNA synthesis, storage, pcr augmentation, and then sequencing, and the
    errors that can occur in each step. returns a sample of DNA and the original strands.

    :param n_orig_strands: number of original strands
    :param length: length of original strands
    :param physical_redundancy: number of times each strand will be duplicated in the synthesis
    :param p_sub_syn: probability for substitution in the synthesis phase
    :param p_del_syn: probability for deletion in the synthesis phase
    :param p_ins_syn: probability for insertion in the synthesis phase
    :param p_term_max_syn: maximum probability for termination in the synthesis phase
    :param p_decay: a list of decay rate for each symbol in the storage phase
    :param pcr_rounds: number of PCR rounds
    :param n_final_samples: number of final samples to be drawn
    :param p_sub_seq: probability for substitution in the sequencing phase
    :param ends_factor_seq: multiplicative factor for sequencing errors in the ends of the strand

    """
    if p_decay is None:
        p_decay = [1e-3, 3e-3, 5e-3, 7e-3]
    # original strands sampling
    orig_strands = sample_orig_strands(n_orig_strands, length)
    # synthesis
    samples = synthesis(
        orig_strands,
        physical_redundancy=physical_redundancy,
        p_sub=p_sub_syn,
        p_del=p_del_syn,
        p_ins=p_ins_syn,
        p_term_max=p_term_max_syn
    )
    # PCR amplification
    for i in range(pcr_rounds):
        samples = pcr(samples)
    # storage
    samples = storage(samples, p_decay)
    # sampling
    samples = sampling(samples, n_final_samples)
    # sequencing
    samples = sequencing(samples, p_sub_seq, ends_factor_seq)

    return DNASampleSet(orig_strands, samples)


if __name__ == "__main__":
    start = time.time()
    # dna_samples = create_dna_samples(
    #     n_orig_strands=100,
    #     length=150,
    #     physical_redundancy=1000,
    #     p_sub_syn=0.02,
    #     p_del_syn=0.01,
    #     p_ins_syn=0.005,
    #     p_term_max_syn=1e-3,
    #     p_decay=[1e-3, 2e-3, 3e-3, 4e-3],
    #     pcr_rounds=3,
    #     n_final_samples=100 * 100,
    #     p_sub_seq=5e-3,
    #     ends_factor_seq=3,
    # )
    dna_samples = create_dna_samples()
    end = time.time()
    print(f"time: {end - start}")
    print(len(dna_samples.samples))
    print(len(dna_samples.orig_strands))
    print(dna_samples.samples[10].strand)
    print(dna_samples.orig_strands[dna_samples.samples[10].orig_idx])
