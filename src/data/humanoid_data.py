"""Humanoid data loading module."""

import numpy as np
from pathlib import Path
from typing import Tuple
from .base_datamodule import BaseClassifierDataModule


class HumanoidDataModule(BaseClassifierDataModule):
    """DataModule for Humanoid system."""
    
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
        """Load Humanoid data.
        
        Args:
            data_file: Path to roa_labels.txt or roa_labels_normalized.txt (ignored if data_dir is set)
            train_size: Number of samples for training (not used for humanoid)
            val_size: Number of samples for validation (not used for humanoid)
            
        Returns:
            Tuple of (X_train, y_train, X_val, y_val, feature_max)
        """
        # Use shared data directory if specified
        if self.data_dir_override:
            data_dir = Path(self.data_dir_override)
            # Try roa_labels_normalized.txt first, then roa_labels.txt
            labels_file = data_dir / "roa_labels_normalized.txt"
            if not labels_file.exists():
                labels_file = data_dir / "roa_labels.txt"
        else:
            labels_file = Path(data_file)
        
        # Load data
        data = np.loadtxt(labels_file, delimiter=',')
        
        # Separate features and labels
        features = data[:, :-1].astype(np.float32)  # All columns except last
        labels = data[:, -1].astype(np.int64)  # Last column
        
        # Split into train/val
        total_samples = len(features)
        train_size_actual = int(total_samples * 0.8)  # 80% train, 20% val
        
        X_train = features[:train_size_actual]
        y_train = labels[:train_size_actual]
        X_val = features[train_size_actual:]
        y_val = labels[train_size_actual:]
        
        # Normalization (if data is already normalized, feature_max will be ~1.0)
        feature_max = np.abs(X_train).max(axis=0)
        # Avoid division by zero
        feature_max = np.where(feature_max == 0, 1.0, feature_max)
        
        # Normalize
        X_train = X_train / feature_max
        X_val = X_val / feature_max
        
        return X_train, y_train, X_val, y_val, feature_max


