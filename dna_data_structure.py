"""Here are the basic data structure for the dna samples"""
from collections import defaultdict
from pathlib import Path
import pickle
import json

import config


A = 0
C = 1
G = 2
T = 3

symbol_int_to_char = ['A', 'C', 'G', 'T']
symbol_char_to_int = {char: i for i, char in enumerate(symbol_int_to_char)}


def int_list_to_string(int_list: list[int]) -> str:
    """Converts a numeric symbol list to a string of ACGT."""
    s = map(symbol_int_to_char.__getitem__, int_list)
    return ''.join(s)


def string_to_int_list(s: str) -> list[int]:
    """Converts a string to an integer list."""
    l = map(symbol_char_to_int.get, s)
    return list(l)


class DNASample:
    """A class that represents a DNA samples with potential errors.

    Attributes:
          strand: the sequence of symbols
          orig_idx: the index of the original strand
    """
    def __init__(self, strand: list[int], orig_idx: int):
        self.strand = strand
        self.orig_idx = orig_idx


class DNASampleSet:
    """A class that represents the DNA sample, with the physical reads that we got relating to the original strands.

    Attributes:
          orig_strands: the list of the original strands
          samples: the list od the strands observed with potential errors, with index relating to the original
            strand

    """
    def __init__(self, orig_strands: list[list[int]], samples: list[DNASample]):
        self.orig_strands = orig_strands
        self.samples = samples

    def to_dict(self) -> dict:
        """This can be used for saving the data in .json format
        will be structured like:
            {"original_strands": ["ACGT...", "CGT...",...],
            "sample_strands": ["ACGT...",...],
            "sample_original_index": [1, 11, 10,...],
            }
        """
        d = defaultdict(list)
        for orig_strand in self.orig_strands:
            strand_string = int_list_to_string(orig_strand)
            d['original_strands'].append(strand_string)
        for sample in self.samples:
            sample_strand = int_list_to_string(sample.strand)
            d['sample_strands'].append(sample_strand)
            d['sample_original_indices'].append(sample.orig_idx)
        return d

    def to_json(self, json_file: Path):
        """Saves the object to a json file."""
        with json_file.open('w') as jf:
            json.dump(self.to_dict(), jf)

    @classmethod
    def from_dict(cls, d: dict):
        """Returns a DNASampleSet from a dictionary constructed as described in "to_dict" method."""
        orig_strands = map(string_to_int_list, d['original_strands'])
        orig_strands = list(orig_strands)
        samples = []
        for i in range(len(d['sample_strands'])):
            strand = string_to_int_list(d['sample_strands'][i])
            orig_idx = d['sample_original_indices'][i]
            samples.append(DNASample(strand=strand, orig_idx=orig_idx))

        return cls(orig_strands=orig_strands, samples=samples)

    @classmethod
    def from_json(cls, json_file: Path):
        """Reads the object from a saved json file"""
        with json_file.open('r') as jf:
            d = json.load(jf)
        return cls.from_dict(d)


def convert_from_pickle_to_json(pickle_file: Path):
    """Converts a created dataset from pickle to json file."""
    with pickle_file.open('rb') as pf:
        dna_sample_set: DNASampleSet = pickle.load(pf)
    json_file = pickle_file.with_suffix('.json')
    dna_sample_set.to_json(json_file)


def convert_all_pickle_to_json():
    """converts all the pickle files in data directory to json"""
    for pickle_file in config.DATA_DIR.glob('*.pkl'):
        print(f"converting {pickle_file}...")
        convert_from_pickle_to_json(pickle_file)


if __name__ == "__main__":
    convert_all_pickle_to_json()
