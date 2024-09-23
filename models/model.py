# models/fusion_net.py

import torch
import torch.nn as nn
from .ct_stream import CTStream
from .tc_stream import TCStream, compute_cwt_batch

class FusionNet(nn.Module):
    """
    FusionNet model that combines the CT Stream and TC Stream.

    Args:
        config (Config): Configuration object containing model hyperparameters.
    """
    def __init__(self, config):
        super(FusionNet, self).__init__()
        self.ct_stream = CTStream(config)
        self.tc_stream = TCStream(config)
        self.num_classes = config.num_classes

        # Fusion layer to combine outputs from both streams
        self.fusion_layer = nn.Linear(self.num_classes * 2, self.num_classes)

    def forward(self, x: torch.Tensor, seq_lengths: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the FusionNet model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_dim)
            seq_lengths (torch.Tensor): Sequence lengths tensor of shape (batch_size,)

        Returns:
            torch.Tensor: Output logits tensor of shape (batch_size, num_classes)
        """
        # CT Stream forward pass
        logits_ct = self.ct_stream(x, seq_lengths)  # Shape: (batch_size, num_classes)

        # Compute CWT for TC Stream
        x_cwt = compute_cwt_batch(x, seq_lengths, config)  # Shape: (batch_size, input_dim, seq_length, scales)

        # TC Stream forward pass
        logits_tc = self.tc_stream(x_cwt)  # Shape: (batch_size, num_classes)

        # Concatenate logits from both streams
        combined_logits = torch.cat([logits_ct, logits_tc], dim=1)  # Shape: (batch_size, num_classes * 2)

        # Fusion layer
        output = self.fusion_layer(combined_logits)  # Shape: (batch_size, num_classes)

        return output
