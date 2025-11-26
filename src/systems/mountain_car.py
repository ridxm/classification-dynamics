"""Mountain Car system definition."""

from typing import List, Tuple
from .base import BaseSystem


class MountainCarSystem(BaseSystem):
    """Mountain Car dynamical system."""
    
    @property
    def state_dim(self) -> int:
        return 2  # [position, velocity]
    
    @property
    def action_dim(self) -> int:
        return 1  # Force applied
    
    @property
    def state_bounds(self) -> List[Tuple[float, float]]:
        """State bounds: position [-2.0, 1.0], velocity [-0.1, 0.1]"""
        return [(-2.0, 1.0), (-0.1, 0.1)]


