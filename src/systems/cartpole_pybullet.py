"""CartPole PyBullet system definition."""

from typing import List, Tuple
from .base import BaseSystem


class CartPolePyBulletSystem(BaseSystem):
    """CartPole PyBullet dynamical system."""
    
    @property
    def state_dim(self) -> int:
        return 4  # [x, theta, x_dot, theta_dot]
    
    @property
    def action_dim(self) -> int:
        return 1  # Force applied to cart
    
    @property
    def state_bounds(self) -> List[Tuple[float, float]]:
        """State bounds based on dataset description."""
        # From dataset_description.json:
        # x: [-6.0, 6.0] m
        # theta: [-π, π] rad
        # x_dot: [-5.0, 5.0] m/s (initial sampling, can exceed)
        # theta_dot: [-5.0, 5.0] rad/s (initial sampling, can exceed)
        return [
            (-6.0, 6.0),      # x (cart position)
            (-3.14159, 3.14159),  # theta (pole angle)
            (-5.0, 5.0),     # x_dot (cart velocity)
            (-5.0, 5.0),     # theta_dot (pole angular velocity)
        ]


