"""This is the pytorch wrapper for the DNASampleSet to use it for training the model."""
import numpy as np

from create_data import load_data_to_dna_sample_set

PAD_SYMBOL = -1
SEQUENCE_LEN = 120
N_SYMBOLS = 4


def pad_list_to_numpy(list_vector: list[int], target_length: int = SEQUENCE_LEN, pad_value: int = PAD_SYMBOL):
    """Adds padding to the end of the list / or cuts the sequence"""
    pad_width = target_length - len(list_vector)
    if pad_width < 0:
        return np.array(list_vector[:target_length])
    return np.pad(list_vector, (0, pad_width), mode='constant', constant_values=pad_value)


def to_one_hot_encoding_plains(vector: np.ndarray, n_symbols: int = N_SYMBOLS):
    """Transforms a sequence of integers of shape (len,) to 2 dim array, having shape of (n_symbols, len)"""
    n = len(vector)
    one_hot_plains = np.zeros((n_symbols, n), dtype=np.float32)
    for i in range(n_symbols):
        mask = vector == i
        one_hot_plains[i, mask] = 1
    return one_hot_plains


def process_strand(strand: list[int]) -> np.ndarray:
    """Combines padding and one-hot-encoding in a single operation."""
    strand = pad_list_to_numpy(strand)
    return to_one_hot_encoding_plains(strand)


def load_data_to_numpy(part: str, generator: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """loads the data to a form that is easy to use for training the model."""
    dna_sample_set = load_data_to_dna_sample_set(part, generator)
    n_samples = len(dna_sample_set.samples)
    # initialize the empty arrays
    noisy_one_hot_encoded = np.empty((n_samples, N_SYMBOLS, SEQUENCE_LEN), dtype=np.float32)
    original_one_hot_encoded = np.empty((n_samples, N_SYMBOLS, SEQUENCE_LEN), dtype=np.float32)
    cluster_indices = np.empty(n_samples, dtype=np.int32)

    for i, dna_sample in enumerate(dna_sample_set.samples):
        cluster_indices[i] = dna_sample.orig_idx

        noisy = dna_sample.strand
        noisy_one_hot_encoded[i] = process_strand(noisy)

        original = dna_sample_set.orig_strands[dna_sample.orig_idx]
        original_one_hot_encoded[i] = process_strand(original)

    return noisy_one_hot_encoded, original_one_hot_encoded, cluster_indices


class DNADataset:
    """A simple interface to the dataset"""
    def __init__(self, part: str, generator: str):
        self.noisy_one_hot_encoded, self.original_one_hot_encoded, self.cluster_indices = \
            load_data_to_numpy(part, generator)
        self.n_samples = len(self.noisy_one_hot_encoded)

    def __len__(self):
        return self.n_samples

    def iter_batches(self, batch_size: int, shuffle: bool, return_cluster: bool):
        indices = np.arange(self.n_samples)
        if shuffle:
            rng = np.random.default_rng()
            rng.shuffle(indices)
        for start in range(0, self.n_samples, batch_size):
            end = min(self.n_samples, start + batch_size)
            if end == start + 1:
                continue  # skip the batch if it is only size 1
            batch_indices = indices[start:end]
            noisy = self.noisy_one_hot_encoded[batch_indices]
            original = self.original_one_hot_encoded[batch_indices]
            if return_cluster:
                clusters = self.cluster_indices[batch_indices]
                yield noisy, original, clusters
            else:
                yield noisy, original


def load_datasets(generator: str):
    train = DNADataset('train', generator)
    dev = DNADataset('dev', generator)
    test = DNADataset('test', generator)
    return train, dev, test


def example():
    import time
    start = time.time()
    train_set, dev_set, test_set = load_datasets('complex')
    end = time.time()
    print(f"took {end - start:.3f} seconds to load the data.")

    start = time.time()
    n_batches = 0
    for noisy, orig in train_set.iter_batches(batch_size=32, shuffle=True, return_cluster=False):
        assert noisy.shape[1:] == (N_SYMBOLS, SEQUENCE_LEN)
        assert orig.shape[1:] == (N_SYMBOLS, SEQUENCE_LEN)
        n_batches += 1
    end = time.time()
    print(f"took {end-start:.3f} seconds to iterate the train set, n_batches={n_batches}, "
          f"avg={(end-start)/n_batches:.3f}")

    start = time.time()
    n_batches = 0
    for noisy, orig, clusters in dev_set.iter_batches(batch_size=32, shuffle=False, return_cluster=True):
        n_batches += 1
    end = time.time()
    print(f"took {end - start:.3f} seconds to iterate the dev set, n_batches={n_batches}, "
          f"avg={(end - start) / n_batches:.3f}")

    start = time.time()
    n_batches = 0
    for noisy, orig, clusters in test_set.iter_batches(batch_size=32, shuffle=False, return_cluster=True):
        n_batches += 1
    end = time.time()
    print(f"took {end - start:.3f} seconds to iterate the test set, n_batches={n_batches}, "
          f"avg={(end - start) / n_batches:.3f}")


if __name__ == "__main__":
    example()
