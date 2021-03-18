"""There will be 3 x 2 partitions, (train, dev, test) x (simple, complex)"""
from complex_generative_model import DNASampleSet
from complex_generative_model import create_dna_samples as generate_complex
from simple_generative_model import generate_dna_sample_set as generate_simple

import pickle

import config

import logging
logging.basicConfig(level=logging.INFO)


def save_file(o, file):
    with open(file, 'wb') as fd:
        pickle.dump(o, fd)


def load_file(file):
    with open(file, 'rb') as fd:
        o = pickle.load(fd)
    return o


def get_data_file_name(part, generator):
    return f"{generator}-{part}-data.pkl"


def create_data():
    # create the different datasets
    partitions = ['train', 'test', 'dev']
    for part in partitions:
        logging.info(f"{part}...")
        # simple generative model
        logging.info("simple...")
        dna_sample_set = generate_simple(**config.simple_data_parameters)
        filename = get_data_file_name(part, 'simple')
        save_file(dna_sample_set, filename)
        # complex generative model
        logging.info('complex...')
        dna_sample_set = generate_complex(**config.complex_data_parameters)
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
