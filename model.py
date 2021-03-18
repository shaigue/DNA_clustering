"""The definition of the DNN model that we use.
This should output both the embedding, and the final output, given a fixed size , 1d input
"""
import torch
from torch import nn
from torch.nn import functional as F


def get_basic_conv(kernel_size: int, n_channels: int, ):
    assert kernel_size % 2 == 1, f"kernel size should be odd. got {kernel_size}"
    padding = kernel_size // 2
    return nn.Conv1d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                     padding=padding, bias=False)


class Encoder(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.n_channels = 4
        self.conv1 = get_basic_conv(kernel_size, self.n_channels)
        self.conv2 = get_basic_conv(kernel_size, self.n_channels)

    def forward(self, x):
        batch_size, n_channels, seq_len = x.shape
        assert self.n_channels == n_channels, f"bad number of channels, should be {self.n_channels} got {n_channels}"
        assert seq_len % 2 == 0, f"sequence length should be even, got {seq_len}"
        x = self.conv1(x)
        x = torch.relu(x)
        x = torch.avg_pool1d(x, kernel_size=2)
        x = self.conv2(x)
        x = torch.tanh(x)
        return x


class Decoder(nn.Module):
    def __init__(self, kernel_size: int):
        super(Decoder, self).__init__()
        self.n_channels = 4
        self.conv1 = get_basic_conv(kernel_size, self.n_channels)
        self.conv2 = get_basic_conv(kernel_size, self.n_channels)

    def forward(self, x):
        batch_size, n_channels, seq_len = x.shape
        assert self.n_channels == n_channels, f"bad number of channels, should be {self.n_channels} got {n_channels}"
        x = self.conv1(x)
        x = torch.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=False)
        x = self.conv2(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        kernel_size = 5
        self.encoder = Encoder(kernel_size)
        self.decoder = Decoder(kernel_size)

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        x = self.decoder(z)
        return torch.flatten(z, start_dim=1), x


def reconstruction_loss(original, reconstructed):
    n_samples = original.shape[0]
    return torch.norm(original - reconstructed) / n_samples


def k_means_loss(sample_embedding, centroid_embedding):
    n_samples = sample_embedding.shape[0]
    return torch.norm(sample_embedding - centroid_embedding) / n_samples


def combined_loss(embedding, original, reconstructed):
    n_samples = embedding.shape[0]
    assert n_samples % 2 == 0, f"embedding should be of even size. got {n_samples}"
    n_samples = n_samples // 2
    k_means_loss_coef = 0.75
    reconstruction_loss_coef = 1 - k_means_loss_coef
    sample_embedding, centroid_embedding = embedding[:n_samples], embedding[n_samples:]
    return k_means_loss_coef * k_means_loss(sample_embedding, centroid_embedding) + \
           reconstruction_loss_coef * reconstruction_loss(original, reconstructed)


def main():
    # make sure that the input is divisible by 2
    # if s=1, then p=k//2
    x = torch.randn(16, 108, 4)
    x = torch.transpose(x, 1, 2)
    model = AutoEncoder()
    embedding, reconstruction = model(x)
    print(embedding.shape)
    print(reconstruction.shape)


if __name__ == "__main__":
    main()
