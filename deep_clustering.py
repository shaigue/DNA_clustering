import numpy as np
from torch.utils.data import Dataset

from create_dataset import load_data
from model import AutoEncoder
from trainer import Trainer


PAD_SYMBOL = -1


def pad_list_to_numpy(list_vector: list[int], target_length: int, pad_value: int):
    pad_width = target_length - len(list_vector)
    assert pad_width >= 0, f"vector length={len(list_vector)} is larger then target_length={target_length}"
    return np.pad(list_vector, (0, pad_width), mode='constant', constant_values=pad_value)


def to_one_hot_encoding_plains(vector: np.ndarray, n_symbols: int):
    n = len(vector)
    one_hot_plains = np.zeros((n_symbols, n), dtype=np.float32)
    for i in range(4):
        mask = vector == i
        one_hot_plains[i, mask] = 1
    return one_hot_plains


def load_data_to_numpy(partition, generator):
    # data is a tuple with (error_strand, original_strand)
    # it should be transformed to one-hot-encoding, and with dimensions
    dna_sample_set = load_data(partition, generator)
    # create a pair for each error strand his original strand
    data = []
    cluster_index = []
    for dna_sample in dna_sample_set.err_strands:
        error = dna_sample.strand
        original = dna_sample_set.orig_strands[dna_sample.orig_idx]
        data.append((error, original))
        cluster_index.append(dna_sample.orig_idx)

    max_sequence_len = 0
    for error, original in data:
        max_sequence_len = max(len(error), len(original), max_sequence_len)
    # make it divisible by 2
    max_sequence_len += max_sequence_len % 2
    # pad all the sequences and convert to numpy
    padded = []
    for error, original in data:
        error = pad_list_to_numpy(error, target_length=max_sequence_len, pad_value=PAD_SYMBOL)
        original = pad_list_to_numpy(original, target_length=max_sequence_len, pad_value=PAD_SYMBOL)
        padded.append((error, original))
    # one hot encode them
    one_hot = []
    for error, original in padded:
        error = to_one_hot_encoding_plains(error, 4)
        original = to_one_hot_encoding_plains(original, 4)
        one_hot.append((error, original))
    return one_hot, cluster_index


class DNADataset(Dataset):
    def __init__(self, partition: str, generator: str, return_cluster_label: bool = False):
        super(DNADataset, self).__init__()
        self.one_hot_encoded, self.cluster_index = load_data_to_numpy(partition, generator)
        self.return_cluster_label = return_cluster_label

    def __getitem__(self, index):
        if self.return_cluster_label:
            return self.one_hot_encoded[index], self.cluster_index[index]
        return self.one_hot_encoded[index]

    def __len__(self):
        return len(self.one_hot_encoded)


def run_training():
    train_dataset = DNADataset('train', 'complex')
    dev_dataset = DNADataset('dev', 'complex', return_cluster_label=True)
    model = AutoEncoder()
    trainer = Trainer(model, train_dataset, dev_dataset)
    # trainer.train(3)
    accuracy = trainer.evaluate()
    print(accuracy)


if __name__ == "__main__":
    run_training()


