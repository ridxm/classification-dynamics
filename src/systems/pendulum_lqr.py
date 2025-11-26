"""Pendulum LQR system definition."""

from typing import List, Tuple
from .base import BaseSystem


class PendulumLQRSystem(BaseSystem):
    """Pendulum with LQR controller system."""
    
    @property
    def state_dim(self) -> int:
        return 2  # [theta, theta_dot] (raw state)
    
    @property
    def action_dim(self) -> int:
        return 1  # Torque applied
    
    @property
    def state_bounds(self) -> List[Tuple[float, float]]:
        """State bounds: theta [-pi, pi], theta_dot [-inf, inf] (approximate)"""
        return [(-3.14159, 3.14159), (-10.0, 10.0)]  # Approximate bounds


