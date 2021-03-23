"""There will be 3 x 2 partitions, (train, dev, test) x (simple, complex).
train will be  big, with many strands and different clusters,

dev and test will be relatively small with ~200 clusters and ~20 items in each one.
"""
import pickle

from complex_generative_model import DNASampleSet
from complex_generative_model import create_dna_samples as generate_complex
from simple_generative_model import generate_dna_sample_set as generate_simple

import logging
logging.basicConfig(level=logging.INFO)


N_TRAIN_CLUSTERS = 250_000
N_TEST_CLUSTERS = 200
CLUSTER_SIZE = 10
STRAND_LENGTH = 100


def save_file(o, file):
    with open(file, 'wb') as fd:
        pickle.dump(o, fd)


def load_file(file):
    with open(file, 'rb') as fd:
        o = pickle.load(fd)
    return o


def get_data_file_name(part, generator):
    return f"data/{generator}-{part}-data.pkl"


def create_data():
    # create the different datasets
    partitions = ['train', 'test', 'dev']
    for part in partitions:
        logging.info(f"{part}...")
        if part == 'train':
            n_clusters = N_TRAIN_CLUSTERS
            pcr_rounds = 2
        else:
            n_clusters = N_TEST_CLUSTERS
            pcr_rounds = 3

        # simple generative model
        logging.info("simple...")
        dna_sample_set = generate_simple(
            n_clusters=n_clusters,
            cluster_size=CLUSTER_SIZE,
            error_p=0.04,
            strand_length=STRAND_LENGTH,
        )
        filename = get_data_file_name(part, 'simple')
        save_file(dna_sample_set, filename)
        # complex generative model
        logging.info('complex...')
        dna_sample_set = generate_complex(
            n_orig_strands=n_clusters,
            length=STRAND_LENGTH,
            physical_redundancy=CLUSTER_SIZE // 2,
            p_sub_syn=0.02,
            p_del_syn=0.01,
            p_ins_syn=0.01,
            p_term_max_syn=0.1,
            p_decay=[0.002, 0.004, 0.006, 0.008],
            pcr_rounds=pcr_rounds,
            n_final_samples=n_clusters * CLUSTER_SIZE,
            p_sub_seq=0.02,
            ends_factor_seq=5,
        )
        filename = get_data_file_name(part, 'complex')
        save_file(dna_sample_set, filename)
    logging.info("finished generating data.")


def load_data(part, generator) -> DNASampleSet:
    filename = get_data_file_name(part, generator)
    return load_file(filename)


if __name__ == "__main__":
    create_data()
    for part in ['train', 'test', 'dev']:
        for generator in ['complex', 'simple']:
            o = load_data(part, generator)
            assert isinstance(o, DNASampleSet)
    logging.info("finished!!!")
