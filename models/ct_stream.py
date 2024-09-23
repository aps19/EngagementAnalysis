# models/ct_stream.py

import torch
import torch.nn as nn
from .conformer import ConformerBlock  # Use relative import


class PositionalEncoding(nn.Module):
    """
    Adds positional encoding to the input embeddings to provide sequence position information.

    Args:
        dim_model (int): Dimension of the embeddings.
        max_len (int): Maximum length of the input sequences.
    """
    def __init__(self, dim_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        # Create a matrix of shape (max_len, dim_model) with positional encodings
        pe = torch.zeros(max_len, dim_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / dim_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd indices
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)  # Register as buffer to avoid updating during training

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encoding to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, dim_model)

        Returns:
            torch.Tensor: Tensor with positional encoding added.
        """
        x = x + self.pe[:, :x.size(1)]
        return x

class CTStream(nn.Module):
    """
    Temporal-Spatial Stream (CT Stream) using Conformer blocks.

    Args:
        config (Config): Configuration object containing model hyperparameters.
    """
    def __init__(self, config):
        super(CTStream, self).__init__()
        self.input_dim = config.input_dim
        self.dim_model = config.dim_model
        self.num_heads = config.num_heads
        self.num_layers = config.num_layers
        self.num_classes = config.num_classes
        self.max_sequence_length = config.max_sequence_length

        # Input projection to model dimension
        self.input_proj = nn.Linear(self.input_dim, self.dim_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(self.dim_model, max_len=self.max_sequence_length)

        # Stack of Conformer blocks
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(self.dim_model, self.num_heads) for _ in range(self.num_layers)
        ])

        # Final classification layer
        self.classifier = nn.Linear(self.dim_model, self.num_classes)

    def forward(self, x: torch.Tensor, seq_lengths: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the CT Stream.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_dim)
            seq_lengths (torch.Tensor): Sequence lengths tensor of shape (batch_size,)

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, num_classes)
        """
        # Input projection
        x = self.input_proj(x)  # Shape: (batch_size, seq_length, dim_model)

        # Add positional encoding
        x = self.positional_encoding(x)

        # Pass through Conformer blocks
        for block in self.conformer_blocks:
            x = block(x)

        # Generate mask to zero out padded positions
        mask = self.generate_mask(seq_lengths, x.size(1), x.device)  # Shape: (batch_size, seq_length)
        x = x * mask.unsqueeze(-1)  # Apply mask

        # Average pooling over valid positions
        sum_x = x.sum(dim=1)  # Sum over sequence length dimension
        avg_x = sum_x / seq_lengths.unsqueeze(-1)  # Divide by actual lengths

        # Classification
        logits = self.classifier(avg_x)  # Shape: (batch_size, num_classes)

        return logits

    @staticmethod
    def generate_mask(seq_lengths: torch.Tensor, max_len: int, device: torch.device) -> torch.Tensor:
        """
        Generates a mask for padded positions.

        Args:
            seq_lengths (torch.Tensor): Sequence lengths tensor.
            max_len (int): Maximum sequence length.
            device (torch.device): Device to place the mask tensor.

        Returns:
            torch.Tensor: Mask tensor of shape (batch_size, seq_length)
        """
        batch_size = seq_lengths.size(0)
        # Create a mask where valid positions are 1 and padded positions are 0
        mask = torch.arange(max_len, device=device).expand(batch_size, max_len) < seq_lengths.unsqueeze(1)
        return mask.float()
