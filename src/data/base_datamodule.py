"""Base PyTorch Lightning DataModule for classifier training."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pytorch_lightning as pl


class TimestepDataset(Dataset):
    """Dataset for individual timestep classification."""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """
        Args:
            features: Array of shape (N, feature_dim)
            labels: Array of shape (N,) with class labels
        """
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).long()
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class BaseClassifierDataModule(pl.LightningDataModule, ABC):
    """Base DataModule for classifier training.
    
    Handles train/val/test splits, DataLoader creation, and class weighting.
    System-specific data loading is handled by load_data_file() method.
    """
    
    def __init__(
        self,
        data_file: str,
        batch_size: int = 256,
        num_workers: int = 4,
        train_size: int = 1000,
        val_size: int = 500,
        use_weighted_sampler: bool = True,
        pin_memory: bool = True,
    ):
        """
        Args:
            data_file: Path to data file (roa_labels.txt or similar)
            batch_size: Batch size for training
            num_workers: Number of DataLoader workers
            train_size: Number of trajectories for training
            val_size: Number of trajectories for validation
            use_weighted_sampler: Whether to use WeightedRandomSampler for class balancing
            pin_memory: Whether to pin memory in DataLoader
        """
        super().__init__()
        self.data_file = data_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_size = train_size
        self.val_size = val_size
        self.use_weighted_sampler = use_weighted_sampler
        self.pin_memory = pin_memory
        
        # Will be set in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.feature_max = None
    
    @abstractmethod
    def load_data_file(
        self,
        data_file: str,
        train_size: int,
        val_size: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load and preprocess data from file.
        
        Args:
            data_file: Path to data file
            train_size: Number of trajectories for training
            val_size: Number of trajectories for validation
            
        Returns:
            Tuple of (X_train, y_train, X_val, y_val, feature_max)
            - X_train: Training features (N_train, feature_dim)
            - y_train: Training labels (N_train,)
            - X_val: Validation features (N_val, feature_dim)
            - y_val: Validation labels (N_val,)
            - feature_max: Max values per feature for normalization (feature_dim,)
        """
        pass
    
    def setup(self, stage: Optional[str] = None):
        """Load data and create datasets."""
        if stage == "test" and self.test_dataset is not None:
            return
        
        # Load data
        X_train, y_train, X_val, y_val, feature_max = self.load_data_file(
            self.data_file,
            self.train_size,
            self.val_size,
        )
        
        self.feature_max = feature_max
        
        # Create datasets
        self.train_dataset = TimestepDataset(X_train, y_train)
        self.val_dataset = TimestepDataset(X_val, y_val)
    
    def train_dataloader(self):
        """Create training DataLoader with optional weighted sampling."""
        if self.use_weighted_sampler:
            # Calculate class weights
            labels = self.train_dataset.labels.numpy()
            unique, counts = np.unique(labels, return_counts=True)
            total = len(labels)
            num_classes = len(unique)
            
            class_weights = {}
            for cls, count in zip(unique, counts):
                weight = total / (num_classes * count)
                class_weights[int(cls)] = weight
            
            sample_weights = np.array(
                [class_weights[int(label)] for label in labels],
                dtype=np.float32
            )
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
            shuffle = False
        else:
            sampler = None
            shuffle = True
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
    
    def val_dataloader(self):
        """Create validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
    
    def test_dataloader(self):
        """Create test DataLoader."""
        if self.test_dataset is None:
            raise ValueError("Test dataset not loaded. Call setup('test') first.")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


