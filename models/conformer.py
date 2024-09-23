# models/conformer.py

import torch
import torch.nn as nn

class ConformerBlock(nn.Module):
    """
    Implementation of a single Conformer block, which combines convolution,
    multi-head self-attention, and feed-forward layers with residual connections.

    Args:
        dim_model (int): Dimension of the model (input and output features).
        num_heads (int): Number of attention heads in the multi-head attention.
        ff_mult (int): Multiplicative factor for the hidden dimension in feed-forward layers.
        conv_kernel_size (int): Kernel size for the convolutional layer.
    """
    def __init__(self, dim_model: int, num_heads: int, ff_mult: int = 4, conv_kernel_size: int = 31):
        super(ConformerBlock, self).__init__()
        self.dim_model = dim_model

        # Feed-Forward Module 1
        self.ffn1 = nn.Sequential(
            nn.LayerNorm(dim_model),
            nn.Linear(dim_model, ff_mult * dim_model),
            nn.GELU(),
            nn.Linear(ff_mult * dim_model, dim_model)
        )

        # Multi-Head Self-Attention Module
        self.mha = nn.MultiheadAttention(embed_dim=dim_model, num_heads=num_heads, batch_first=True)

        # Convolution Module
        self.conv_module = nn.Sequential(
            nn.LayerNorm(dim_model),
            nn.Conv1d(dim_model, dim_model * 2, kernel_size=1),  # Pointwise Conv
            nn.GLU(dim=1),
            nn.Conv1d(dim_model, dim_model, kernel_size=conv_kernel_size, padding=conv_kernel_size // 2, groups=dim_model),
            nn.BatchNorm1d(dim_model),
            nn.ReLU(),
            nn.Conv1d(dim_model, dim_model, kernel_size=1)  # Final Pointwise Conv
        )

        # Feed-Forward Module 2
        self.ffn2 = nn.Sequential(
            nn.LayerNorm(dim_model),
            nn.Linear(dim_model, ff_mult * dim_model),
            nn.GELU(),
            nn.Linear(ff_mult * dim_model, dim_model)
        )

        # Final Layer Normalization
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Conformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, dim_model)

        Returns:
            torch.Tensor: Output tensor of the same shape as input.
        """
        # Save residual for later addition
        residual = x

        # Feed-Forward Module 1 with residual connection
        x = x + 0.5 * self.ffn1(x)

        # Multi-Head Self-Attention Module with residual connection
        attn_output, _ = self.mha(x, x, x)
        x = x + attn_output

        # Convolution Module with residual connection
        x_conv = x.transpose(1, 2)  # Transpose to (batch_size, dim_model, seq_length) for Conv1d
        x_conv = self.conv_module(x_conv)
        x_conv = x_conv.transpose(1, 2)  # Transpose back to (batch_size, seq_length, dim_model)
        x = x + x_conv

        # Feed-Forward Module 2 with residual connection
        x = x + 0.5 * self.ffn2(x)

        # Final Layer Normalization
        x = self.layer_norm(x)

        # Add initial residual connection
        x = x + residual

        return x
