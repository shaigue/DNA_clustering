import time

from collections import defaultdict
import torch
from torch import nn
from sklearn.cluster import KMeans
from torch.utils.data import Dataset, DataLoader

from model import combined_loss
import numpy as np

from clustering_accuracy import convert_cluster_labels_to_partition, clustering_accuracy


class Trainer:
    def __init__(self, model: nn.Module, train_dataset: Dataset, dev_dataset: Dataset, eval_every=5, loss_lambda=0.5,
                 clustering_gamma=0.9, weight_decay=0.01):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        self.optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay)
        self.eval_every = eval_every
        self.loss_lambda = loss_lambda
        self.clustering_gamma = clustering_gamma

    def train(self, n_epochs: int, verbose=True, evaluate=True) -> dict:
        """Train the model for n_epochs"""
        training_log = defaultdict(list)

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
        return dict(training_log)

    def train_epoch(self) -> float:
        """Train on the entire train data once"""
        loss = 0
        for batch in self.train_loader:
            loss += self.train_batch(batch)
        return loss / len(self.train_loader)

    def train_batch(self, batch) -> float:
        """Training iteration on a single batch"""
        self.optimizer.zero_grad()
        sample, centroids = batch
        sample = sample.to(self.device)
        centroids = centroids.to(self.device)

        sample_embedding, decoded_samples = self.model(sample)
        centroid_embedding = self.model(centroids, embedding_only=True)
        
        loss = combined_loss(samples=sample, sample_embedding=sample_embedding,
                             centroid_embedding=centroid_embedding, decoded_samples=decoded_samples,
                             loss_lambda=self.loss_lambda)

        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().numpy()

    def evaluate(self, dataset=None):
        """Evaluates the accuracy of the model over the given dataset."""
        if dataset is None:
            dataset = self.dev_dataset
        loader = DataLoader(dataset, batch_size=64, shuffle=False)

        with torch.no_grad():
            embedding_list = []
            label_list = []
            for (error, original), labels in loader:
                error = error.to(self.device)
                embedding = self.model(error, embedding_only=True)
                embedding_list.append(embedding.cpu().numpy())
                label_list.append(labels)

            embedding_list = np.concatenate(embedding_list)
            label_list = np.concatenate(label_list)
            # assumes that the labels are {0, 1, 2, ... n - 1}. so the number of clusters is 'n'
            n_clusters = np.max(label_list) + 1

            clustering_algo = KMeans(n_clusters)
            cluster_estimation = clustering_algo.fit_predict(embedding_list)

            # convert the clustering to a partition so it will fit the definition of the accuracy defined in the paper
            label_list = convert_cluster_labels_to_partition(label_list)
            cluster_estimation = convert_cluster_labels_to_partition(cluster_estimation)
            accuracy = clustering_accuracy(label_list, cluster_estimation, min_part_coef=self.clustering_gamma)

        return accuracy



