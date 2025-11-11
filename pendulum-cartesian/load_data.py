"""
Load Pendulum Cartesian trajectory data and prepare for training.
Follows the same approach as cartpole:
- Load trajectories and break into individual datapoints
- Each timestep becomes a separate datapoint with the trajectory's label
- Transform theta (first element) to sin(theta) and cos(theta)
- Apply selective normalization (only normalize features that need it)
"""

import numpy as np
import os
from pathlib import Path

LOCAL_DIR = Path(__file__).parent
# Data is in shared directory
SHARED_DATA_DIR = Path("/common/users/shared/pracsys/genMoPlan/data_trajectories/pendulum_cartesian_50k")
TRAJECTORIES_DIR = SHARED_DATA_DIR / "trajectories"

def transform_theta_to_sincos_timestep(timestep):
    """
    Transform a single data point (timestep).
    Input: [theta, ?, ?, ?] (4 features) - first element is theta
    Output: [sin(theta), cos(theta), ?, ?, ?] (5 features)
    """
    theta = timestep[0]  # First element is theta
    feature_2 = timestep[1]
    feature_3 = timestep[2]
    feature_4 = timestep[3]
    
    # Transform theta to sin/cos
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    
    # Return: [sin(theta), cos(theta), feature_2, feature_3, feature_4]
    return np.array([sin_theta, cos_theta, feature_2, feature_3, feature_4], dtype=np.float32)

def load_individual_datapoints(sequence_files, labels, start_index, set_name, traj_dir='trajectories'):
    """
    Load trajectories and break them into individual data points.
    Each timestep becomes a separate datapoint with the trajectory's label.
    
    Args:
        sequence_files: List of sequence filenames
        labels: Array of labels from roa_labels.txt
        start_index: Starting index in shuffled_indices.txt (for label mapping)
        set_name: Name for logging
        traj_dir: Directory containing trajectory files
    
    Returns:
        X: Array of shape (num_datapoints, 5) - individual state vectors [sin(theta), cos(theta), ?, ?, ?]
        y: Array of shape (num_datapoints,) - labels (1 or 0)
    """
    X_datapoints = []
    y_datapoints = []
    
    print(f"\nLoading {set_name} data...")
    print(f"  Processing {len(sequence_files)} trajectories...")
    valid_count = 0
    
    for idx, seq_file in enumerate(sequence_files):
        traj_path = TRAJECTORIES_DIR / seq_file
        
        # Check if file exists
        if not os.path.exists(traj_path):
            continue
        
        # Load trajectory file (each line is a state vector: [theta, ?, ?, ?])
        trajectory = np.loadtxt(traj_path, delimiter=',')
        if trajectory.ndim == 1:
            # Single timestep trajectory
            trajectory = trajectory.reshape(1, -1)
        
        # Get label from roa_labels.txt
        # row_idx in roa_labels.txt corresponds to position in shuffled_indices.txt
        row_idx = start_index + idx
        trajectory_label = labels[row_idx]  # This trajectory's label (1 or 0)
        
        # Break trajectory into individual data points
        # Each timestep becomes a separate datapoint with the trajectory's label
        for timestep in trajectory:
            # Transform: [theta, ?, ?, ?] -> [sin(theta), cos(theta), ?, ?, ?]
            datapoint = transform_theta_to_sincos_timestep(timestep)
            X_datapoints.append(datapoint)
            y_datapoints.append(trajectory_label)  # All timesteps from this trajectory get the same label
        
        valid_count += 1
        if valid_count % 500 == 0:
            print(f"  Processed {valid_count}/{len(sequence_files)} trajectories...")
    
    X_datapoints = np.array(X_datapoints, dtype=np.float32)
    y_datapoints = np.array(y_datapoints, dtype=np.int64)
    
    print(f"✓ Loaded {valid_count} trajectories")
    print(f"  Total individual datapoints: {len(X_datapoints)}")
    print(f"  X shape: {X_datapoints.shape}  (datapoints, features)")
    print(f"  y shape: {y_datapoints.shape}  (labels)")
    print(f"  Success rate: {np.mean(y_datapoints == 1):.2%}")
    print(f"  Failure rate: {np.mean(y_datapoints == 0):.2%}")
    
    return X_datapoints, y_datapoints


def load_pendulum_cartesian_data(train_size=1000, val_size=500):
    """
    Load training and validation data for Pendulum Cartesian.
    
    Args:
        train_size: Number of sequences for training (default: 1000)
        val_size: Number of sequences for validation (default: 500)
    
    Returns:
        (X_train, y_train), (X_val, y_val), feature_max
    """
    print("=" * 80)
    print("LOADING PENDULUM CARTESIAN TRAJECTORY DATA")
    print("=" * 80)
    
    # Load shuffled indices
    shuffled_indices_file = SHARED_DATA_DIR / "shuffled_indices.txt"
    with open(shuffled_indices_file, 'r') as f:
        shuffled_sequences = [line.strip() for line in f.readlines()]
    print(f"\nLoading shuffled indices...")
    print(f"✓ Found {len(shuffled_sequences)} sequences in shuffled order")
    
    # Load labels from roa_labels.txt
    labels_file = SHARED_DATA_DIR / "roa_labels.txt"
    labels_data = np.loadtxt(labels_file, delimiter=',')
    labels = labels_data[:, -1].astype(int) # Last column is the label
    print(f"\nLoading labels from roa_labels.txt...")
    print(f"✓ Found {len(labels)} labels")
    print(f"  Success rate: {np.mean(labels == 1):.2%}")
    print(f"  Failure rate: {np.mean(labels == 0):.2%}")
    
    # Split shuffled sequences into train/val
    train_sequences = shuffled_sequences[:train_size]
    val_sequences = shuffled_sequences[train_size:train_size + val_size]
    
    print(f"\nData split:")
    print(f"  Training: {len(train_sequences)} sequences")
    print(f"  Validation: {len(val_sequences)} sequences")
    
    X_train, y_train = load_individual_datapoints(train_sequences, labels, start_index=0, set_name="training")
    X_val, y_val = load_individual_datapoints(val_sequences, labels, start_index=train_size, set_name="validation")
    
    # Normalize data: divide by max (per feature)
    # Calculate max values from training data (per feature)
    feature_max = np.abs(X_train).max(axis=0)  # Max absolute value per feature
    print(f"\nFeature max values (per feature): {feature_max}")
    print(f"  sin(theta) max: {feature_max[0]:.6f}")
    print(f"  cos(theta) max: {feature_max[1]:.6f}")
    print(f"  Feature 2 max: {feature_max[2]:.6f}")
    print(f"  Feature 3 max: {feature_max[3]:.6f}")
    print(f"  Feature 4 max: {feature_max[4]:.6f}")
    
    # For pendulum-cartesian, we'll normalize features [2], [3], [4] (the non-sin/cos features)
    # Features [0] and [1] are sin(theta) and cos(theta) which are already in [-1, 1]
    # But let's check what the actual features are - for now, normalize all except sin/cos
    # Actually, let's normalize [2], [3], [4] only (similar to cartpole where we normalize [0], [3], [4])
    print(f"\nNormalizing training data...")
    print(f"  Normalizing features [2], [3], [4] only (keeping sin/cos as-is)")
    X_train[:, [2, 3, 4]] = X_train[:, [2, 3, 4]] / feature_max[[2, 3, 4]]
    X_val[:, [2, 3, 4]] = X_val[:, [2, 3, 4]] / feature_max[[2, 3, 4]]
    
    print(f"✓ Training data normalized")
    print(f"✓ Validation data normalized")
    print(f"  X_train normalized range: [{X_train.min():.4f}, {X_train.max():.4f}]")
    print(f"  X_val normalized range: [{X_val.min():.4f}, {X_val.max():.4f}]")
    
    print("\n" + "=" * 80)
    print("DATA SUMMARY")
    print("=" * 80)
    print(f"Training datapoints: {len(X_train)}")
    print(f"Validation datapoints: {len(X_val)}")
    print(f"Feature max values: {feature_max}")
    print(f"Number of features: {X_train.shape[1]}")
    
    return X_train, y_train, X_val, y_val, feature_max


def load_test_data(test_start_index=1000, feature_max=None):
    """
    Load test data directly from roa_labels.txt.
    
    Args:
        test_start_index: Starting row index in roa_labels.txt (default: 1000, after training)
        feature_max: Max values from training data for normalization (required)
    
    Returns:
        (X_test, y_test)
    """
    print("=" * 80)
    print("LOADING TEST DATA")
    print("=" * 80)
    
    if feature_max is None:
        raise ValueError("feature_max must be provided for test data normalization!")
    
    # Load roa_labels.txt - each row is already an individual datapoint
    # Format: [theta, feature_2, feature_3, feature_4, label] (5 columns)
    print(f"\nLoading test data from roa_labels.txt (starting from row {test_start_index})...")
    labels_file = SHARED_DATA_DIR / "roa_labels.txt"
    labels_data = np.loadtxt(labels_file, delimiter=',')
    
    # Get test data (rows test_start_index onwards)
    test_data = labels_data[test_start_index:]
    
    print(f"Total rows in roa_labels.txt: {len(labels_data)}")
    print(f"Test datapoints: {len(test_data)}")
    
    # Extract features and labels
    # Features: columns 0-3 are [theta, feature_2, feature_3, feature_4]
    # Label: column 4 (last column)
    test_features_raw = test_data[:, :4]  # [theta, feature_2, feature_3, feature_4]
    y_test = test_data[:, 4].astype(int)  # Labels
    
    # Transform theta to sin/cos for each datapoint
    print(f"\nTransforming theta to sin/cos...")
    X_test = []
    for timestep in test_features_raw:
        transformed = transform_theta_to_sincos_timestep(timestep)
        X_test.append(transformed)
    X_test = np.array(X_test, dtype=np.float32)
    
    # Normalize test data using the same max values from training
    # Only normalize features [2], [3], [4] (the non-sin/cos features)
    print(f"\nNormalizing test data using training max values...")
    print(f"Feature max values: {feature_max}")
    print(f"Normalizing only features [2], [3], [4] (keeping sin/cos as-is)")
    X_test[:, [2, 3, 4]] = X_test[:, [2, 3, 4]] / feature_max[[2, 3, 4]]
    
    print(f"X_test shape: {X_test.shape}  (datapoints, features)")
    print(f"y_test shape: {y_test.shape}  (labels)")
    print(f"X_test normalized range: [{X_test.min():.4f}, {X_test.max():.4f}]")
    print(f"Success rate: {np.mean(y_test == 1):.2%}")
    print(f"Failure rate: {np.mean(y_test == 0):.2%}")
    
    return X_test, y_test


if __name__ == "__main__":
    X_train, y_train, X_val, y_val, feature_max = load_pendulum_cartesian_data()

