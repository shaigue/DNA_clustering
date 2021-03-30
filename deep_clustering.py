"""Runs multiple training runs and find the best models"""

import json
from itertools import product

import torch

from dna_dataset import load_datasets
from model import AutoEncoder
from train import Trainer


def train_best_models(generator: str):
    """Runs on all many different hyper parameter configurations and saves the best model"""
    train_data, dev_data, test_data = load_datasets(generator)
    loss_lambda_options = [0, 0.1, 0.5, 0.9, 0.99]
    weight_decay_options = [0, 1e-4]
    n_epochs = 30
    logs = []
    best_dev_accuracy = 0
    best_name = None
    for i, (loss_lambda, weight_decay) in enumerate(product(loss_lambda_options, weight_decay_options)):
        name = f"{generator}_{i}"
        model = AutoEncoder()
        trainer = Trainer(name, model, train_data, dev_data, loss_lambda=loss_lambda, weight_decay=weight_decay)
        train_logs = trainer.train(n_epochs)
        dev_accuracy = train_logs['best_accuracy'][-1]
        print(f'name={name}, loss_lambda={loss_lambda}, weight_decay={weight_decay}, dev_accuracy={dev_accuracy}')
        if dev_accuracy > best_dev_accuracy:
            best_dev_accuracy = dev_accuracy
            best_name = name

        logs.append({
            'name': name,
            'loss_lambda': loss_lambda,
            'weight_decay': weight_decay,
            'train_logs': train_logs
        })

    best_model = AutoEncoder()
    state_dict = torch.load(f'{best_name}.pt', map_location='cpu')
    best_model.load_state_dict(state_dict)
    trainer = Trainer(best_name, best_model, train_data, dev_data)
    test_accuracy = trainer.evaluate(test_data)

    print(f"best name={best_name}, test_accuracy={test_accuracy}")
    return {'logs': logs, 'best_name': best_name, 'test_accuracy': test_accuracy}


def main():
    logs = {}
    for generator in ['simple', 'complex']:
        logs[generator] = train_best_models(generator)
    with open('deep_clustering_logs_1.json', 'w') as f:
        json.dump(logs, f)


if __name__ == "__main__":
    main()
