"""Pendulum LQR data loading module."""

import numpy as np
from pathlib import Path
from typing import Tuple
from .base_datamodule import BaseClassifierDataModule


class PendulumLQRDataModule(BaseClassifierDataModule):
    """DataModule for Pendulum LQR system."""
    
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
        """Load Pendulum LQR trajectory data with sin/cos transformation.
        
        Args:
            data_file: Path to roa_labels.txt
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
        labels = labels_data[:, -1].astype(int)
        
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
        
        # Calculate normalization
        feature_max = np.abs(X_train).max(axis=0)
        
        # Normalize
        X_train = X_train / feature_max
        X_val = X_val / feature_max
        
        return X_train, y_train, X_val, y_val, feature_max
    
    def _transform_theta_to_sincos(self, timestep: np.ndarray) -> np.ndarray:
        """Transform [theta, theta_dot] to [sin(theta), cos(theta), theta_dot]."""
        theta = timestep[0]
        theta_dot = timestep[1]
        return np.array([np.sin(theta), np.cos(theta), theta_dot], dtype=np.float32)
    
    def _load_individual_datapoints(
        self,
        sequence_files: list,
        labels: np.ndarray,
        start_index: int,
        trajectories_dir: Path,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load trajectories and break into individual datapoints with transformation."""
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
            
            # Break into datapoints with transformation
            for timestep in trajectory:
                transformed = self._transform_theta_to_sincos(timestep)
                X_datapoints.append(transformed)
                y_datapoints.append(trajectory_label)
        
        return np.array(X_datapoints, dtype=np.float32), np.array(y_datapoints, dtype=np.int64)


