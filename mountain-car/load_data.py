"""
Load Mountain Car trajectory data and prepare for training.
Follows the same approach as cartpole:
- Load trajectories and break into individual datapoints
- Each timestep becomes a separate datapoint with the trajectory's label
- Apply selective normalization (only normalize features that need it)
"""

import numpy as np
import os

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
        X: Array of shape (num_datapoints, 2) - individual state vectors [position, velocity]
        y: Array of shape (num_datapoints,) - labels (1 or 0)
    """
    X_datapoints = []
    y_datapoints = []
    
    print(f"\nLoading {set_name} data...")
    print(f"  Processing {len(sequence_files)} trajectories...")
    valid_count = 0
    
    for idx, seq_file in enumerate(sequence_files):
        traj_path = os.path.join(traj_dir, seq_file)
        
        # Check if file exists
        if not os.path.exists(traj_path):
            continue
        
        # Load trajectory file (each line is a state vector: [position, velocity])
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
            # State: [position, velocity] (2 features)
            datapoint = timestep.astype(np.float32)
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


def load_training_data(train_size=1000, val_size=500, traj_dir='trajectories'):
    """
    Load training and validation data for Mountain Car.
    
    Args:
        train_size: Number of sequences for training (default: 1000)
        val_size: Number of sequences for validation (default: 500)
        traj_dir: Directory containing trajectory files
    
    Returns:
        (X_train, y_train), (X_val, y_val), feature_max
    """
    print("=" * 80)
    print("LOADING MOUNTAIN CAR TRAJECTORY DATA")
    print("=" * 80)
    
    # Load shuffled indices to determine order
    print("\nLoading shuffled indices...")
    with open('shuffled_indices.txt', 'r') as f:
        shuffled_sequences = [line.strip() for line in f.readlines()]
    print(f"✓ Found {len(shuffled_sequences)} sequences in shuffled order")
    
    # Load labels from roa_labels.txt
    # Row N in roa_labels.txt corresponds to line N in shuffled_indices.txt
    print("\nLoading labels from roa_labels.txt...")
    labels_data = np.loadtxt('roa_labels.txt', delimiter=',')
    labels = labels_data[:, -1].astype(int)  # Last column is the label
    print(f"✓ Found {len(labels)} labels")
    print(f"  Success rate: {np.mean(labels == 1):.2%}")
    print(f"  Failure rate: {np.mean(labels == 0):.2%}")
    
    # Extract sequences for training and validation
    train_sequences = shuffled_sequences[:train_size]
    val_sequences = shuffled_sequences[train_size:train_size + val_size]
    
    print(f"\nData split:")
    print(f"  Training: {len(train_sequences)} sequences")
    print(f"  Validation: {len(val_sequences)} sequences")
    
    # Load training set: first train_size sequences -> individual datapoints
    X_train, y_train = load_individual_datapoints(
        train_sequences, labels, start_index=0, set_name="training", traj_dir=traj_dir
    )
    
    # Load validation set: next val_size sequences -> individual datapoints
    X_val, y_val = load_individual_datapoints(
        val_sequences, labels, start_index=train_size, set_name="validation", traj_dir=traj_dir
    )
    
    # Calculate max values from training data for normalization
    # Mountain Car state: [position, velocity]
    # Position range: [-2.0, 1.0]
    # Velocity range: [-0.1, 0.1]
    # We'll normalize both features
    feature_max = np.abs(X_train).max(axis=0)  # Max absolute value per feature
    print(f"\nFeature max values (per feature): {feature_max}")
    print(f"  Position max: {feature_max[0]:.6f}")
    print(f"  Velocity max: {feature_max[1]:.6f}")
    
    # Normalize training data: divide by max (per feature)
    # For mountain car, we normalize both features [0, 1]
    print(f"\nNormalizing training data...")
    X_train = X_train / feature_max
    X_val = X_val / feature_max
    
    print(f"✓ Training data normalized")
    print(f"✓ Validation data normalized")
    print(f"  X_train normalized range: [{X_train.min():.4f}, {X_train.max():.4f}]")
    print(f"  X_val normalized range: [{X_val.min():.4f}, {X_val.max():.4f}]")
    
    return (X_train, y_train), (X_val, y_val), feature_max


def load_test_data(test_start_index=1000, traj_dir='trajectories', feature_max=None):
    """
    Load test data directly from roa_labels.txt.
    
    Args:
        test_start_index: Starting row index in roa_labels.txt (default: 1000, after training)
        traj_dir: Directory containing trajectory files
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
    # Format: [position, velocity, label] (3 columns)
    print(f"\nLoading test data from roa_labels.txt (starting from row {test_start_index})...")
    labels_data = np.loadtxt('roa_labels.txt', delimiter=',')
    
    # Get test data (rows test_start_index onwards)
    test_data = labels_data[test_start_index:]
    
    print(f"Total rows in roa_labels.txt: {len(labels_data)}")
    print(f"Test datapoints: {len(test_data)}")
    
    # Extract features and labels
    # Features: columns 0-1 are [position, velocity]
    # Label: column 2 (last column)
    test_features_raw = test_data[:, :2]  # [position, velocity]
    y_test = test_data[:, 2].astype(int)  # Labels
    
    # Convert to float32
    X_test = test_features_raw.astype(np.float32)
    
    # Normalize test data using the same max values from training
    print(f"\nNormalizing test data using training max values...")
    print(f"Feature max values: {feature_max}")
    X_test = X_test / feature_max
    
    print(f"X_test shape: {X_test.shape}  (datapoints, features)")
    print(f"y_test shape: {y_test.shape}  (labels)")
    print(f"X_test normalized range: [{X_test.min():.4f}, {X_test.max():.4f}]")
    print(f"Success rate: {np.mean(y_test == 1):.2%}")
    print(f"Failure rate: {np.mean(y_test == 0):.2%}")
    
    return X_test, y_test


if __name__ == "__main__":
    # Example usage
    print("Mountain Car Data Loader")
    print("=" * 80)
    
    # Load training and validation data
    (X_train, y_train), (X_val, y_val), feature_max = load_training_data(
        train_size=1000, 
        val_size=500
    )
    
    print("\n" + "=" * 80)
    print("DATA SUMMARY")
    print("=" * 80)
    print(f"Training datapoints: {len(X_train)}")
    print(f"Validation datapoints: {len(X_val)}")
    print(f"Feature max values: {feature_max}")
    print(f"Number of features: {X_train.shape[1]}")
    
    # Optionally load test data
    # X_test, y_test = load_test_data(test_start_index=1000, feature_max=feature_max)

