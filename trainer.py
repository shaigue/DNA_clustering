import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from model import AutoEncoder, combined_loss
import numpy as np

from clustering_accuracy import convert_cluster_labels_to_partition, clustering_accuracy
from sklearn.cluster import KMeans, MiniBatchKMeans, OPTICS


class Trainer:
    def __init__(self, model: nn.Module, train_dataset: Dataset, dev_dataset: Dataset):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.dev_dataset.return_cluster_label = True
        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        self.dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False)
        self.optimizer = torch.optim.Adam(model.parameters())

    def train(self, n_epochs=1):
        for i in range(n_epochs):
            loss = self.train_epoch()
            print(f'epoch={i + 1}, mean_loss={loss}')

    def train_epoch(self) -> float:
        loss = 0
        for batch in self.train_loader:
            loss += self.train_batch(batch)
        return loss / len(self.train_loader)

    def train_batch(self, batch) -> float:
        self.optimizer.zero_grad()
        error, original = batch
        error = error.to(self.device)
        original = original.to(self.device)
        concatenated = torch.cat((error, original))
        embedding, reconstructed = self.model(concatenated)
        loss = combined_loss(embedding, concatenated, reconstructed)
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().numpy()

    def evaluate(self):
        all_embedding = []
        all_labels = []
        for (error, original), labels in self.dev_loader:
            error = error.to(self.device)
            embedding, _ = self.model(error)
            all_embedding.append(embedding.detach().cpu().numpy())
            all_labels.append(labels)
        all_embedding = np.concatenate(all_embedding)
        all_labels = np.concatenate(all_labels)
        n_clusters = np.max(all_labels) + 1
        all_labels = convert_cluster_labels_to_partition(all_labels)
        clustering_algo = KMeans(n_clusters)
        # clustering_algo = OPTICS()
        # clustering_algo = MiniBatchKMeans(n_clusters)

        cluster_estimation = clustering_algo.fit_predict(all_embedding)
        cluster_estimation = convert_cluster_labels_to_partition(cluster_estimation)
        accuracy = clustering_accuracy(all_labels, cluster_estimation, min_part_coef=0.9)

        return accuracy


