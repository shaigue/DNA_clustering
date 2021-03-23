"""The definition of the DNN model that we use.
This should output both the embedding, and the final output, given a fixed size , 1d input
"""
import torch
from torch import nn
from torch.nn import functional as F


def get_basic_conv(kernel_size: int, n_channels: int, out_channels: int = None):
    assert kernel_size % 2 == 1, f"kernel size should be odd. got {kernel_size}"
    padding = kernel_size // 2
    if out_channels is None:
        out_channels = n_channels
    return nn.Conv1d(in_channels=n_channels, out_channels=out_channels, kernel_size=kernel_size,
                     padding=padding, bias=False)


class Encoder(nn.Module):
    """This is the encoder module that takes a padded one-hot-encoded DNA strand and returns his encoding"""
    def __init__(self, kernel_size: int, n_channels: int):
        super().__init__()
        self.n_channels = n_channels
        self.conv1 = get_basic_conv(kernel_size, self.n_channels)
        self.conv2 = get_basic_conv(kernel_size, self.n_channels)
        self.conv3 = get_basic_conv(kernel_size, self.n_channels, 1)

    def forward(self, x):
        batch_size, n_channels, seq_len = x.shape
        assert self.n_channels == n_channels, f"bad number of channels, should be {self.n_channels} got {n_channels}"
        assert seq_len % 2 == 0, f"sequence length should be even, got {seq_len}"
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.avg_pool1d(x, kernel_size=2)
        x = self.conv3(x)
        # remove the channel dimension
        x = torch.reshape(x, (batch_size, seq_len // 2))
        x = torch.tanh(x)
        return x


class Decoder(nn.Module):
    """This is the decoder Module that takes the embedding of the encoder and tries to reconstruct the inputs."""
    def __init__(self, kernel_size: int, n_channels: int):
        super(Decoder, self).__init__()
        self.n_channels = n_channels
        self.conv1 = get_basic_conv(kernel_size, 1, self.n_channels)
        self.conv2 = get_basic_conv(kernel_size, self.n_channels)
        self.conv3 = get_basic_conv(kernel_size, self.n_channels)

    def forward(self, x):
        batch_size, seq_len_half = x.shape
        # add the channel dimension so can use convolution on it
        x = torch.reshape(x, (batch_size, 1, seq_len_half))
        x = self.conv1(x)
        x = torch.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=False)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        kernel_size = 5
        n_channels = 4
        self.encoder = Encoder(kernel_size, n_channels)
        self.decoder = Decoder(kernel_size, n_channels)

    def forward(self, x: torch.Tensor, embedding_only=False):
        """
        :param embedding_only: for inference, we don't have to calculate the reconstruction.
        :return:
        """
        z = self.encoder(x)
        if embedding_only:
            return z
        x = self.decoder(z)
        return z, x


def reconstruction_loss(samples, decoded_samples):
    """A loss that takes into account the difference between the input to the encoder and the output of the decoder,
    and forces them to be small."""
    return torch.norm(samples - decoded_samples)


def k_means_loss(sample_embedding, centroid_embedding):
    """A loss that takes into account the difference between the embedding of the error sample and the original sample,
    and forces them to be small."""
    return torch.norm(sample_embedding - centroid_embedding)


def combined_loss(samples, sample_embedding, centroid_embedding, decoded_samples, loss_lambda):
    """The loss combining the auxiliary loss and the clustering loss."""
    assert 0 <= loss_lambda <= 1, f"factor should be between 0 and 1, got {loss_lambda}"
    aux_loss = reconstruction_loss(samples, decoded_samples)
    cluster_loss = k_means_loss(sample_embedding, centroid_embedding)
    return loss_lambda * cluster_loss + (1 - loss_lambda) * aux_loss


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
