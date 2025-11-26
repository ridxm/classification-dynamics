"""CartPole PyBullet data loading module."""

import numpy as np
from pathlib import Path
from typing import Tuple
from .base_datamodule import BaseClassifierDataModule


class CartPolePyBulletDataModule(BaseClassifierDataModule):
    """DataModule for CartPole PyBullet system."""
    
    def __init__(self, *args, data_dir: str = None, **kwargs):
        """Initialize with optional data_dir override."""
        super().__init__(*args, **kwargs)
        self.data_dir_override = data_dir
    
    def load_data_file(
        self,
        data_file: str,
        train_size: int,
        val_size: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load CartPole PyBullet trajectory data.
        
        Transforms theta to sin/cos: [x, theta, x_dot, theta_dot] -> [x, sin(theta), cos(theta), x_dot, theta_dot]
        Only normalizes features [0, 3, 4] (x, x_dot, theta_dot), NOT [1, 2] (sin/cos).
        
        Args:
            data_file: Path to roa_labels.txt (ignored if data_dir is set)
            train_size: Number of trajectories for training
            val_size: Number of trajectories for validation
            
        Returns:
            Tuple of (X_train, y_train, X_val, y_val, feature_max)
        """
        # Use shared data directory if specified, otherwise use local
        if self.data_dir_override:
            data_dir = Path(self.data_dir_override)
        else:
            data_dir = Path(data_file).parent
        
        trajectories_dir = data_dir / "trajectories"
        shuffled_indices_file = data_dir / "shuffled_indices.txt"
        labels_file = data_dir / "roa_labels.txt" if self.data_dir_override else Path(data_file)
        
        # Load shuffled indices
        with open(shuffled_indices_file, 'r') as f:
            shuffled_sequences = [line.strip() for line in f.readlines()]
        
        # Load labels
        labels_data = np.loadtxt(labels_file, delimiter=',')
        labels = labels_data[:, -1].astype(int)  # Last column is the label
        
        # Split sequences
        train_sequences = shuffled_sequences[:train_size]
        val_sequences = shuffled_sequences[train_size:train_size + val_size]
        
        # Load training data
        X_train, y_train = self._load_individual_datapoints(
            train_sequences, labels, start_index=0, trajectories_dir=trajectories_dir
        )
        
        # Load validation data
        X_val, y_val = self._load_individual_datapoints(
            val_sequences, labels, start_index=train_size, trajectories_dir=trajectories_dir
        )
        
        # Calculate normalization (only for features [0, 3, 4])
        feature_max = np.abs(X_train).max(axis=0)
        # Avoid division by zero
        feature_max = np.where(feature_max == 0, 1.0, feature_max)
        
        # Normalize only features [0, 3, 4] (x, x_dot, theta_dot)
        # Do NOT normalize features [1, 2] (sin(theta), cos(theta))
        X_train[:, [0, 3, 4]] = X_train[:, [0, 3, 4]] / feature_max[[0, 3, 4]]
        X_val[:, [0, 3, 4]] = X_val[:, [0, 3, 4]] / feature_max[[0, 3, 4]]
        
        return X_train, y_train, X_val, y_val, feature_max
    
    def _transform_theta_to_sincos_timestep(self, timestep: np.ndarray) -> np.ndarray:
        """Transform a single data point (timestep).
        
        Input: [x, theta, x_dot, theta_dot] (4 features)
        Output: [x, sin(theta), cos(theta), x_dot, theta_dot] (5 features)
        """
        x = timestep[0]
        theta = timestep[1]
        x_dot = timestep[2]
        theta_dot = timestep[3]
        
        # Transform theta to sin/cos
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        # Return: [x, sin(theta), cos(theta), x_dot, theta_dot]
        return np.array([x, sin_theta, cos_theta, x_dot, theta_dot], dtype=np.float32)
    
    def _load_individual_datapoints(
        self,
        sequence_files: list,
        labels: np.ndarray,
        start_index: int,
        trajectories_dir: Path,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load trajectories and break into individual datapoints.
        
        Transforms each timestep: [x, theta, x_dot, theta_dot] -> [x, sin(theta), cos(theta), x_dot, theta_dot]
        """
        X_datapoints = []
        y_datapoints = []
        
        for idx, seq_file in enumerate(sequence_files):
            traj_path = trajectories_dir / seq_file
            
            if not traj_path.exists():
                continue
            
            # Load trajectory
            trajectory = np.loadtxt(traj_path, delimiter=',')
            if trajectory.ndim == 1:
                trajectory = trajectory.reshape(1, -1)
            
            # Get label
            row_idx = start_index + idx
            trajectory_label = labels[row_idx]
            
            # Break into datapoints and transform
            # Raw features: [x, theta, x_dot, theta_dot]
            # Transformed: [x, sin(theta), cos(theta), x_dot, theta_dot]
            for timestep in trajectory:
                transformed_timestep = self._transform_theta_to_sincos_timestep(timestep.astype(np.float32))
                X_datapoints.append(transformed_timestep)
                y_datapoints.append(trajectory_label)
        
        return np.array(X_datapoints, dtype=np.float32), np.array(y_datapoints, dtype=np.int64)


