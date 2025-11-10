"""
Load and preprocess pendulum-cart trajectory data.
Reads from local directory and saves processed data locally.
Uses shuffled_indices.txt to determine order and splits into train/val sets.
Maps each sequence to its label from roa_labels.txt (last column only).
"""

import numpy as np
import os
from pathlib import Path

# Paths
LOCAL_DIR = Path(__file__).parent

def get_sequence_id(filename):
    """Extract numeric ID from sequence filename."""
    # Extract number from "sequence_XXXXX.txt"
    return int(filename.replace("sequence_", "").replace(".txt", ""))

def load_trajectory_data(train_size=1000, val_size=500):
    """
    Load trajectory files and labels.
    Uses shuffled_indices.txt to determine order and splits data.
    Maps line N of shuffled_indices.txt to row N of roa_labels.txt.
    
    Args:
        train_size: Number of sequences for training (default: 1000)
        val_size: Number of sequences for validation (default: 500)
    
    Returns:
        (X_train, y_train, ids_train), (X_val, y_val, ids_val)
    """
    
    print(f"Loading data from {LOCAL_DIR}...")
    
    # Load shuffled indices to determine order
    shuffled_indices_file = LOCAL_DIR / "shuffled_indices.txt"
    print(f"Loading shuffled indices from {shuffled_indices_file}...")
    with open(shuffled_indices_file, 'r') as f:
        shuffled_sequences = [line.strip() for line in f.readlines()]
    
    print(f"Found {len(shuffled_sequences)} sequences in shuffled order")
    
    # Load labels (roa_labels.txt - last column is the label)
    labels_file = LOCAL_DIR / "roa_labels.txt"
    print(f"Loading labels from {labels_file}...")
    labels_data = np.loadtxt(labels_file, delimiter=',')
    
    # Extract labels (last column) - labels[i] corresponds to row i (0-indexed)
    labels = labels_data[:, -1].astype(int)
    
    print(f"Found {len(labels)} labels")
    print(f"  Success rate: {np.mean(labels == 1):.2%}")
    print(f"  Failure rate: {np.mean(labels == 0):.2%}")
    
    # Split shuffled sequences into train/val
    train_sequences = shuffled_sequences[:train_size]
    val_sequences = shuffled_sequences[train_size:train_size + val_size]
    
    print(f"\nData split:")
    print(f"  Training: {len(train_sequences)} sequences")
    print(f"  Validation: {len(val_sequences)} sequences")
    
    # Get trajectory directory
    traj_dir = LOCAL_DIR / "trajectories"
    
    # Find maximum trajectory length and feature dimensions
    print("\nFinding maximum trajectory length...")
    max_timesteps = 0
    num_features = None
    
    # Check a few trajectories to determine dimensions
    for seq_file in shuffled_sequences[:100]:  # Sample first 100
        traj_path = traj_dir / seq_file
        if traj_path.exists():
            traj = np.loadtxt(traj_path, delimiter=',')
            if traj.ndim == 1:
                traj = traj.reshape(1, -1)
            max_timesteps = max(max_timesteps, traj.shape[0])
            if num_features is None:
                num_features = traj.shape[1]
    
    # Check all trajectories to find true max
    print("Scanning all trajectories for maximum length...")
    for seq_file in shuffled_sequences:
        traj_path = traj_dir / seq_file
        if traj_path.exists():
            traj = np.loadtxt(traj_path, delimiter=',')
            if traj.ndim == 1:
                traj = traj.reshape(1, -1)
            max_timesteps = max(max_timesteps, traj.shape[0])
    
    print(f"Maximum trajectory length: {max_timesteps} timesteps")
    print(f"Each trajectory has {num_features} features")
    print(f"Features: x, sin(theta), cos(theta), x_dot, theta_dot")
    
    def load_and_transform_set(sequence_files, set_name, start_index):
        """Load and transform a set of trajectories.
        
        Args:
            sequence_files: List of sequence filenames
            set_name: Name of the set (for logging)
            start_index: Starting index in shuffled_indices.txt (0-indexed)
        """
        if len(sequence_files) == 0:
            return None, None, None
        
        # First, filter to only sequences that exist
        valid_sequences = []
        valid_indices = []
        for idx, seq_file in enumerate(sequence_files):
            traj_path = traj_dir / seq_file
            if traj_path.exists():
                valid_sequences.append(seq_file)
                valid_indices.append(start_index + idx)
            else:
                print(f"  Warning: {seq_file} not found, skipping...")
        
        num_seqs = len(valid_sequences)
        if num_seqs == 0:
            return None, None, None
        
        # Initialize arrays
        # Features: x, sin(theta), cos(theta), x_dot, theta_dot (already transformed)
        X = np.zeros((num_seqs, max_timesteps, num_features), dtype=np.float32)
        y = np.zeros(num_seqs, dtype=int)
        ids = []
        
        print(f"\nLoading {set_name} trajectories...")
        print(f"  Found {num_seqs} valid trajectories out of {len(sequence_files)} listed")
        
        for i, (seq_file, row_idx) in enumerate(zip(valid_sequences, valid_indices)):
            if (i + 1) % 500 == 0:
                print(f"  Processed {i + 1}/{num_seqs} trajectories...")
            
            traj_path = traj_dir / seq_file
            
            # Load trajectory
            trajectory = np.loadtxt(traj_path, delimiter=',')
            
            # Handle 1D case (single timestep)
            if trajectory.ndim == 1:
                trajectory = trajectory.reshape(1, -1)
            
            traj_length = trajectory.shape[0]
            
            # Store trajectory (shorter trajectories will have zeros at the end)
            X[i, :traj_length, :] = trajectory
            
            # Get label from roa_labels.txt row (row_idx corresponds to position in shuffled_indices.txt)
            # row_idx is 0-indexed, so labels[row_idx] is the label for this sequence
            y[i] = labels[row_idx]
            
            # Store sequence ID for reference
            seq_id = get_sequence_id(seq_file)
            ids.append(str(seq_id))
        
        ids = np.array(ids, dtype='<U10')
        
        print(f"✓ Loaded {num_seqs} {set_name} trajectories")
        print(f"  X shape: {X.shape}  (trajectories, time_steps, features)")
        print(f"  y shape: {y.shape}  (labels)")
        print(f"  Success rate: {np.mean(y == 1):.2%}")
        print(f"  Failure rate: {np.mean(y == 0):.2%}")
        
        return X, y, ids
    
    # Load train/val sets
    # Training: indices 0 to train_size-1
    X_train, y_train, ids_train = load_and_transform_set(train_sequences, "training", start_index=0)
    # Validation: indices train_size to train_size+val_size-1
    X_val, y_val, ids_val = load_and_transform_set(val_sequences, "validation", start_index=train_size)
    
    return (X_train, y_train, ids_train), (X_val, y_val, ids_val)

def calculate_class_weights(y):
    """
    Calculate class weights for balanced training.
    Returns weights that can be used with WeightedRandomSampler or in loss function.
    
    Args:
        y: Array of labels (0 or 1)
    
    Returns:
        class_weights: Dictionary with weights for each class
        sample_weights: Array of weights for each sample (for WeightedRandomSampler)
    """
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    
    # Calculate inverse frequency weights
    # Weight for class i = total_samples / (num_classes * count_of_class_i)
    num_classes = len(unique)
    class_weights = {}
    for cls, count in zip(unique, counts):
        weight = total / (num_classes * count)
        class_weights[int(cls)] = weight
    
    # Create sample weights array (weight for each sample based on its class)
    sample_weights = np.array([class_weights[int(label)] for label in y], dtype=np.float32)
    
    return class_weights, sample_weights

def load_test_data_from_roa_labels(test_start_index=1000, test_size=None):
    """
    Load test data directly from roa_labels.txt starting from test_start_index.
    Row N in roa_labels.txt corresponds to line N in shuffled_indices.txt.
    
    Args:
        test_start_index: Starting row index in roa_labels.txt (default: 1000, after training)
        test_size: Number of test samples to load (None = load all remaining)
    
    Returns:
        (X_test, y_test, ids_test)
    """
    print(f"Loading test data from roa_labels.txt (starting from row {test_start_index})...")
    
    # Load roa_labels.txt - this is our source of truth for testing
    labels_file = LOCAL_DIR / "roa_labels.txt"
    labels_data = np.loadtxt(labels_file, delimiter=',')
    
    # Get test rows from roa_labels.txt
    if test_size is None:
        test_labels_data = labels_data[test_start_index:]
    else:
        test_labels_data = labels_data[test_start_index:test_start_index + test_size]
    
    # Extract labels (last column)
    y_test = test_labels_data[:, -1].astype(int)
    
    print(f"Found {len(y_test)} test samples from roa_labels.txt (rows {test_start_index} to {test_start_index + len(y_test) - 1})")
    
    # Now we need to get the corresponding trajectory sequences
    # Row N in roa_labels.txt corresponds to line N in shuffled_indices.txt
    shuffled_indices_file = LOCAL_DIR / "shuffled_indices.txt"
    with open(shuffled_indices_file, 'r') as f:
        shuffled_sequences = [line.strip() for line in f.readlines()]
    
    # Get corresponding sequence filenames
    if test_size is None:
        test_sequences = shuffled_sequences[test_start_index:]
    else:
        test_sequences = shuffled_sequences[test_start_index:test_start_index + test_size]
    
    # Get trajectory directory
    traj_dir = LOCAL_DIR / "trajectories"
    max_timesteps = 502  # Known from training data
    num_features = 4  # Known from training data
    
    # Load trajectory sequences
    valid_sequences = []
    valid_indices = []
    for idx, seq_file in enumerate(test_sequences):
        traj_path = traj_dir / seq_file
        if traj_path.exists():
            valid_sequences.append(seq_file)
            valid_indices.append(test_start_index + idx)
        else:
            print(f"  Warning: {seq_file} not found, skipping...")
    
    num_seqs = len(valid_sequences)
    X_test = np.zeros((num_seqs, max_timesteps, num_features), dtype=np.float32)
    y_test_valid = np.zeros(num_seqs, dtype=int)
    ids_test = []
    
    print(f"Loading {num_seqs} test trajectories...")
    for i, (seq_file, row_idx) in enumerate(zip(valid_sequences, valid_indices)):
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{num_seqs} trajectories...")
        
        traj_path = traj_dir / seq_file
        trajectory = np.loadtxt(traj_path, delimiter=',')
        if trajectory.ndim == 1:
            trajectory = trajectory.reshape(1, -1)
        
        traj_length = trajectory.shape[0]
        X_test[i, :traj_length, :] = trajectory
        
        # Get label from roa_labels.txt row (row_idx)
        y_test_valid[i] = labels_data[row_idx, -1].astype(int)
        
        seq_id = get_sequence_id(seq_file)
        ids_test.append(str(seq_id))
    
    ids_test = np.array(ids_test, dtype='<U10')
    print(f"✓ Loaded {num_seqs} test trajectories")
    print(f"  X shape: {X_test.shape}")
    print(f"  y shape: {y_test_valid.shape}")
    print(f"  Success rate: {np.mean(y_test_valid == 1):.2%}")
    print(f"  Failure rate: {np.mean(y_test_valid == 0):.2%}")
    
    return X_test, y_test_valid, ids_test

def save_processed_data(train_data, val_data, test_data=None,
                       train_filename="trajectory_data_train.npz",
                       val_filename="trajectory_data_val.npz",
                       test_filename="trajectory_data_test.npz"):
    """Save processed data to local directory."""
    X_train, y_train, ids_train = train_data
    X_val, y_val, ids_val = val_data
    
    # Save training set
    if X_train is not None:
        train_path = LOCAL_DIR / train_filename
        print(f"\nSaving training data to {train_path}...")
        np.savez(train_path, X=X_train, y=y_train, ids=ids_train)
        print(f"✓ Saved to {train_path}")
    
    # Save validation set
    if X_val is not None:
        val_path = LOCAL_DIR / val_filename
        print(f"Saving validation data to {val_path}...")
        np.savez(val_path, X=X_val, y=y_val, ids=ids_val)
        print(f"✓ Saved to {val_path}")
    
    # Save test set
    if test_data is not None:
        X_test, y_test, ids_test = test_data
        test_path = LOCAL_DIR / test_filename
        print(f"Saving test data to {test_path}...")
        np.savez(test_path, X=X_test, y=y_test, ids=ids_test)
        print(f"✓ Saved to {test_path}")

if __name__ == "__main__":
    train_data, val_data = load_trajectory_data()
    save_processed_data(train_data, val_data)
    print("\n✓ Data loading complete!")

