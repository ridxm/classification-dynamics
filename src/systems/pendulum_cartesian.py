"""Pendulum Cartesian system definition."""

from typing import List, Tuple
from .base import BaseSystem


class PendulumCartesianSystem(BaseSystem):
    """Pendulum in Cartesian coordinates system."""
    
    @property
    def state_dim(self) -> int:
        return 4  # [x, y, x_dot, y_dot] or similar
    
    @property
    def action_dim(self) -> int:
        return 1  # Torque applied
    
    @property
    def state_bounds(self) -> List[Tuple[float, float]]:
        """State bounds (approximate)"""
        return [(-2.0, 2.0), (-2.0, 2.0), (-5.0, 5.0), (-5.0, 5.0)]


