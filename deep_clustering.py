import numpy as np
from torch.utils.data import Dataset
import torch

from dataset import load_data
from model import AutoEncoder
from train import Trainer

import json
import logging
from itertools import product
from collections import defaultdict

PAD_SYMBOL = -1
BEST_LOSS_LAMBDA = {
    'simple': 0.0, # or 0.01
    'complex': 0.1, # or 0.01
}


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


def load_datasets(generator: str):
    train = DNADataset('train', generator)
    dev = DNADataset('dev', generator, return_cluster_label=True)
    test = DNADataset('test', generator, return_cluster_label=True)
    return train, dev, test


def run_training(generator: str):
    train_data, dev_data, test_data = load_datasets(generator)
    model = AutoEncoder()
    trainer = Trainer(model, train_data, dev_data, loss_lambda=BEST_LOSS_LAMBDA[generator])
    trainer.train(n_epochs=40)
    accuracy = trainer.evaluate(test_data)
    torch.save(model.state_dict(), f'{generator}_trained_model.pt')
    print(accuracy)


def tune_loss_lambda():
    """Runs over different values of the hyper parameters and checks which one of them is the best."""
    logging.basicConfig(filename='loss_lambda_tune.log', filemode='w', level=logging.INFO)
    logging.info("starting loss_lambda_tune.")
    options = [0, 0.01, 0.1, 0.5, 0.9, 0.99, 1]
    logs = {}

    for generator in ['complex', 'simple']:
        train_data, dev_data, _ = load_datasets(generator)
        best_loss_lambda = 0
        best_accuracy = 0
        logs[generator] = {}

        for loss_lambda in options:
            model = AutoEncoder()
            trainer = Trainer(model, train_data, dev_data, loss_lambda=loss_lambda, weight_decay=0)
            train_logs = trainer.train(n_epochs=15, verbose=False, evaluate=True)
            accuracy = trainer.evaluate()

            logging.info(f"lambda={loss_lambda}, accuracy={accuracy}")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_loss_lambda = loss_lambda

            logs[generator][loss_lambda] = {'train_logs': train_logs, 'accuracy': accuracy}

        logs[generator]['best_loss_lambda'] = best_loss_lambda
        logs[generator]['best_accuracy'] = best_accuracy
        logging.info(f"best param={best_loss_lambda}, best accuracy={best_accuracy}")

    with open('loss_lambda_tune.json', 'w') as f:
        json.dump(logs, f)


def tune_complex_hp():
    logging.basicConfig(filename='complex_hp_tune.log', filemode='w', level=logging.INFO)
    logging.info("starting complex hp tune.")

    weight_decay_opt = [0, 1e-4, 1e-2]
    loss_lambda_opt = [0, 0.01, 0.1]
    options = product(weight_decay_opt, loss_lambda_opt)

    train_data, dev_data, _ = load_datasets('complex')
    logs = {}
    best_params = None
    best_accuracy = 0
    for i, (weight_decay, loss_lambda) in enumerate(options):
        model = AutoEncoder()
        trainer = Trainer(model, train_data, dev_data, loss_lambda=loss_lambda, weight_decay=weight_decay)
        train_logs = trainer.train(n_epochs=10, verbose=False, evaluate=True)
        accuracy = trainer.evaluate()

        logs[i] = {'train_logs': train_logs, 'weight_decay': weight_decay, 'loss_lambda': loss_lambda,
                   'accuracy': accuracy}
        logging.info(f"weight_decay={weight_decay}, loss_lambda={loss_lambda}, accuracy={accuracy}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = {'weight_decay': weight_decay, 'loss_lambda': loss_lambda}

    logging.info(f"best param={best_params}, best accuracy={best_accuracy}")
    logs['best_params'] = best_params
    logs['best_accuracy'] = best_accuracy
    with open('complex_hp_tune.json', 'w') as f:
        json.dump(logs, f)
    logging.info("finished.")


if __name__ == "__main__":
    # print("tuning complex...")
    # tune_loss_lambda('complex')
    # print("tuning simple...")
    # tune_loss_lambda('simple')
    # tune_complex_hp()
    tune_loss_lambda()
    # run_training()


