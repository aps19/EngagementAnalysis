# models/tc_stream.py

import torch
import torch.nn as nn
import numpy as np
import pywt

class TCStream(nn.Module):
    """
    Temporal-Frequency Stream (TC Stream) using Continuous Wavelet Transform (CWT) and CNNs.

    Args:
        config (Config): Configuration object containing model hyperparameters.
    """
    def __init__(self, config):
        super(TCStream, self).__init__()
        self.input_dim = config.input_dim
        self.num_classes = config.num_classes
        self.scales = config.scales

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # Output shape: (batch_size, 64, seq_length/2, scales/2)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # Output shape: (batch_size, 128, seq_length/4, scales/4)
        )

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # Output shape: (batch_size, 128, 1, 1)

        # Fully connected layer for classification
        self.fc = nn.Linear(128, self.num_classes)

    def forward(self, x_cwt: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the TC Stream.

        Args:
            x_cwt (torch.Tensor): CWT-transformed input of shape (batch_size, input_dim, seq_length, scales)

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, num_classes)
        """
        # Convolutional layers
        x = self.conv_layers(x_cwt)

        # Global average pooling
        x = self.global_pool(x)

        # Flatten the tensor
        x = x.view(x.size(0), -1)  # Shape: (batch_size, 128)

        # Classification
        logits = self.fc(x)  # Shape: (batch_size, num_classes)

        return logits

def compute_cwt_batch(
    features_batch: torch.Tensor,
    seq_lengths: torch.Tensor,
    config
) -> torch.Tensor:
    """
    Computes the CWT for a batch of feature sequences with variable lengths.

    Args:
        features_batch (torch.Tensor): Input features of shape (batch_size, seq_length, input_dim)
        seq_lengths (torch.Tensor): Sequence lengths for each sample
        config (Config): Configuration object containing model hyperparameters

    Returns:
        torch.Tensor: Time-frequency representation of shape (batch_size, input_dim, seq_length, scales)
    """
    batch_size, max_seq_length, input_dim = features_batch.size()
    scales = np.arange(1, config.scales + 1)
    x_cwt = torch.zeros(batch_size, input_dim, max_seq_length, config.scales, device=features_batch.device)

    for i in range(batch_size):
        seq_len = seq_lengths[i].item()
        for j in range(input_dim):
            # Extract the signal for current feature and sequence length
            signal = features_batch[i, :seq_len, j].cpu().numpy()

            # Compute CWT coefficients
            coeffs, _ = pywt.cwt(signal, scales, 'morl')

            # Convert to tensor and place on appropriate device
            coeffs_tensor = torch.tensor(coeffs.T, device=features_batch.device)  # Shape: (seq_len, scales)

            # Place the coefficients in the output tensor
            x_cwt[i, j, :seq_len, :] = coeffs_tensor

    return x_cwt
