"""There will be 3 x 2 partitions, (train, dev, test) x (simple, complex).
train will be  big, with many strands and different clusters,

dev and test will be relatively small with ~200 clusters and ~20 items in each one.
"""
import json
from pathlib import Path

import config
from dna_data_structure import DNASampleSet
from complex_generative_model import create_dna_samples as generate_complex
from simple_generative_model import generate_dna_sample_set as generate_simple


N_TRAIN_CLUSTERS = 250_000
N_TEST_CLUSTERS = 200
CLUSTER_SIZE = 10
STRAND_LENGTH = 100


def get_data_path(part: str, generator: str) -> Path:
    """Returns the path to the save location"""
    assert part in ['train', 'dev', 'test']
    assert generator in ['complex', 'simple']
    return config.DATA_DIR / f"{generator}-{part}-data.json"


def save_data_to_json(dna_sample_set: DNASampleSet, part: str, generator: str):
    """Saves the sample set to json"""
    data_path = get_data_path(part, generator)
    dna_sample_set.to_json(data_path)


def load_data_to_dict(part: str, generator: str) -> dict:
    """Loads the data into a dictionary"""
    assert part in ['train', 'dev', 'test']
    assert generator in ['complex', 'simple']
    data_path = get_data_path(part, generator)
    with data_path.open('r') as jf:
        d = json.load(jf)
    return d


def load_data_to_dna_sample_set(part: str, generator: str) -> DNASampleSet:
    """Loads the data into a dna sample set"""
    assert part in ['train', 'dev', 'test']
    assert generator in ['complex', 'simple']
    data_path = get_data_path(part, generator)
    return DNASampleSet.from_json(data_path)


def create_data():
    """Generate the data for the experiment."""
    # create the different datasets
    partitions = ['train', 'test', 'dev']
    for part in partitions:
        print(f"creating part={part}...")
        if part == 'train':
            n_clusters = N_TRAIN_CLUSTERS
            pcr_rounds = 2
        else:
            n_clusters = N_TEST_CLUSTERS
            pcr_rounds = 3

        # simple generative model
        print("generating simple data...")
        generator = 'simple'
        dna_sample_set = generate_simple(
            n_clusters=n_clusters,
            cluster_size=CLUSTER_SIZE,
            error_p=0.04,
            strand_length=STRAND_LENGTH,
        )
        save_data_to_json(dna_sample_set, part, generator)
        # complex generative model
        print('generating complex data...')
        generator = 'complex'
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
        save_data_to_json(dna_sample_set, part, generator)
    print("finished generating data.")


def create_original_paper_data():
    """Generate the data for the experiment, using the parameters of the original paper"""
    # create the different datasets
    partitions = ['test', 'dev']
    for part in partitions:
        # simple generative model
        print("generating data...")
        dna_sample_set = generate_simple(
            n_clusters=1_000,
            cluster_size=10,
            error_p=0.04,
            strand_length=110,
        )
        json_file = Path(f'data/original_paper_data_{part}.json')
        dna_sample_set.to_json(json_file)

    print("finished generating data.")


def example():
    for part in ['train', 'dev', 'test']:
        for generator in ['complex', 'simple']:
            try:
                d = load_data_to_dict(part, generator)
                print(f"success loading part={part}, generator={generator}")
            except:
                print(f"fail loading part={part}, generator={generator}")


if __name__ == "__main__":
    # create_data()
    create_original_paper_data()