"""Abstract base class for dynamical systems."""

from abc import ABC, abstractmethod
from typing import Tuple, List
import numpy as np


class BaseSystem(ABC):
    """Abstract base class for dynamical systems.
    
    Each system must define its state dimension, action dimension,
    and state bounds for normalization and visualization.
    """
    
    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Dimension of the state space."""
        pass
    
    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Dimension of the action space."""
        pass
    
    @property
    @abstractmethod
    def state_bounds(self) -> List[Tuple[float, float]]:
        """State bounds for each dimension.
        
        Returns:
            List of (min, max) tuples for each state dimension.
        """
        pass
    
    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state to [0, 1] range based on state bounds.
        
        Args:
            state: State vector of shape (..., state_dim)
            
        Returns:
            Normalized state vector
        """
        state = np.asarray(state)
        bounds = np.array(self.state_bounds)
        
        # Normalize: (state - min) / (max - min)
        normalized = (state - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
        return normalized
    
    def denormalize_state(self, normalized_state: np.ndarray) -> np.ndarray:
        """Denormalize state from [0, 1] range back to original bounds.
        
        Args:
            normalized_state: Normalized state vector of shape (..., state_dim)
            
        Returns:
            Denormalized state vector
        """
        normalized_state = np.asarray(normalized_state)
        bounds = np.array(self.state_bounds)
        
        # Denormalize: normalized * (max - min) + min
        denormalized = normalized_state * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        return denormalized


