"""Main classifier model for timestep-level classification."""

import torch.nn as nn
from .mlp import MLP


class Classifier(nn.Module):
    """Feedforward classifier for individual timestep classification.
    
    This is a wrapper around MLP that provides a consistent interface
    for all systems.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        output_dim: int = 2,
        dropout: float = 0.3,
        use_batch_norm: bool = True,
    ):
        """
        Args:
            input_dim: Input feature dimension (state_dim or transformed state_dim)
            hidden_dims: List of hidden layer dimensions, e.g., [64, 128, 64]
            output_dim: Number of classes (default: 2 for binary classification)
            dropout: Dropout probability
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        self.mlp = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
        )
    
    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Logits tensor of shape (batch_size, output_dim)
        """
        return self.mlp(x)


