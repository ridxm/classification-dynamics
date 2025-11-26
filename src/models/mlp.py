"""MLP building blocks for classifiers."""

import torch
import torch.nn as nn


class MLPBlock(nn.Module):
    """A single MLP block with Linear, BatchNorm, ReLU, and Dropout."""
    
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.3):
        """
        Args:
            in_dim: Input dimension
            out_dim: Output dimension
            dropout: Dropout probability
        """
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class MLP(nn.Module):
    """Multi-layer perceptron with configurable architecture."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        output_dim: int,
        dropout: float = 0.3,
        use_batch_norm: bool = True,
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            dropout: Dropout probability
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        layers = []
        dims = [input_dim] + hidden_dims
        
        # Build hidden layers
        for i in range(len(dims) - 1):
            if use_batch_norm:
                layers.append(MLPBlock(dims[i], dims[i + 1], dropout))
            else:
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        
        self.hidden_layers = nn.Sequential(*layers)
        
        # Output layer (no activation, no dropout)
        self.output_layer = nn.Linear(dims[-1], output_dim)
    
    def forward(self, x):
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x


