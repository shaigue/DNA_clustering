"""Here is the training code for the model."""

from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans
import torch
from torch import nn
from torch import optim

from clustering_accuracy import convert_cluster_labels_to_partition, clustering_accuracy
from dna_dataset import DNADataset
from model import combined_loss


class Trainer:
    """A class to train the model, with the given hyper parameters"""
    def __init__(self, name: str, model: nn.Module, train_dataset: DNADataset, dev_dataset: DNADataset, eval_every=5,
                 loss_lambda=0.5, clustering_gamma=0.9, weight_decay=1e-4, batch_size=64):

        self.name = name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.optimizer = optim.Adam(model.parameters(), weight_decay=weight_decay)
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.9)
        self.eval_every = eval_every
        self.loss_lambda = loss_lambda
        self.clustering_gamma = clustering_gamma
        self.batch_size = batch_size

    def train(self, n_epochs: int, verbose=True, evaluate=True) -> dict:
        """Train the model for n_epochs"""
        training_log = defaultdict(list)
        best_accuracy = 0
        for i in range(n_epochs):
            loss = self.train_epoch()
            training_log['training_loss'].append({'epoch': i, 'value': loss})
            if verbose:
                print(f'epoch={i + 1}, mean_loss={loss}')
            if evaluate and (i + 1) % self.eval_every == 0:
                if verbose:
                    print(f'evaluating...')
                dev_accuracy = self.evaluate(self.dev_dataset)
                training_log['dev_accuracy'].append({'epoch': i, 'value': dev_accuracy})
                if verbose:
                    print(f'dev_accuracy={dev_accuracy}')
                if best_accuracy < dev_accuracy:
                    print(f"saving best model in epoch {i}")
                    best_accuracy = dev_accuracy
                    torch.save(self.model.state_dict(), f"{self.name}.pt")

        training_log['best_accuracy'].append(best_accuracy)
        if best_accuracy > 0:
            state_dict = torch.load(f"{self.name}.pt")
            self.model.load_state_dict(state_dict)

        return dict(training_log)

    def train_epoch(self) -> float:
        """Train on the entire train data once"""
        self.model.train()
        loss = 0
        n_batches = 0
        batch_iter = self.train_dataset.iter_batches(
            batch_size=self.batch_size,
            shuffle=True,
            return_cluster=False
        )
        for batch in batch_iter:
            n_batches += 1
            loss += self.train_batch(batch)
        self.lr_scheduler.step()
        return loss / n_batches

    def to_tensor(self, np_array: np.ndarray) -> torch.Tensor:
        """Converts it to a torch tensor and places it onto the correct device."""
        return torch.tensor(np_array, device=self.device)

    def train_batch(self, batch) -> float:
        """Training iteration on a single batch"""
        self.optimizer.zero_grad()
        sample, centroids = batch
        sample = self.to_tensor(sample)
        centroids = self.to_tensor(centroids)

        sample_embedding, decoded_samples = self.model(sample)
        centroid_embedding = self.model(centroids, embedding_only=True)
        
        loss = combined_loss(samples=sample, sample_embedding=sample_embedding,
                             centroid_embedding=centroid_embedding, decoded_samples=decoded_samples,
                             loss_lambda=self.loss_lambda)

        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().numpy()

    def evaluate(self, dataset: DNADataset = None):
        """Evaluates the accuracy of the model over the given dataset."""
        self.model.eval()
        if dataset is None:
            dataset = self.dev_dataset
        batch_iter = dataset.iter_batches(self.batch_size, shuffle=False, return_cluster=False)

        with torch.no_grad():
            embedding_list = []
            cluster_labels = dataset.cluster_indices
            for error, original in batch_iter:
                error = self.to_tensor(error)
                embedding = self.model(error, embedding_only=True)
                embedding_list.append(embedding.cpu().numpy())

            embedding_list = np.concatenate(embedding_list)
            # assumes that the labels are {0, 1, 2, ... n - 1}. so the number of clusters is 'n'
            n_clusters = np.max(cluster_labels) + 1

            clustering_algo = KMeans(n_clusters)
            cluster_estimation = clustering_algo.fit_predict(embedding_list)

            # convert the clustering to a partition so it will fit the definition of the accuracy defined in the paper
            true_partition = convert_cluster_labels_to_partition(cluster_labels)
            estimated_partition = convert_cluster_labels_to_partition(cluster_estimation)
            accuracy = clustering_accuracy(true_partition, estimated_partition, min_part_coef=self.clustering_gamma)

        return accuracy


def example():
    from dna_dataset import load_datasets
    from model import AutoEncoder
    import time

    train_data, dev_data, test_data = load_datasets('complex')
    model = AutoEncoder()
    trainer = Trainer('example', model, train_data, dev_data)
    start = time.time()
    logs = trainer.train(10)
    end = time.time()
    print(f"10 epoch training took={end-start:.3f} seconds")
    print(logs)


if __name__ == "__main__":
    example()
