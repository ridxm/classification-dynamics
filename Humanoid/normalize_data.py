import numpy as np

def normalize_data(input_file, output_file):
    """
    Normalize the data using min-max normalization.
    Each feature will be scaled to [-1, 1] range.
    """
    print(f"Loading data from {input_file}...")
    
    # Load data
    data = []
    labels = []
    
    with open(input_file, 'r') as f:
        for line in f:
            values = [float(x) for x in line.strip().split(',')]
            # Last value is the label, first 67 are features
            features = values[:-1]
            label = values[-1]
            data.append(features)
            labels.append(label)
    
    # Convert to numpy array
    data = np.array(data)
    labels = np.array(labels)
    
    print(f"Loaded {len(data)} samples with {data.shape[1]} features")
    print(f"Original data range: min={data.min():.6f}, max={data.max():.6f}")
    
    # Apply min-max normalization: (x - min) / (max - min)
    # Calculate min and max for each feature
    feature_mins = data.min(axis=0)
    feature_maxs = data.max(axis=0)
    
    # Avoid division by zero (if max == min, set normalized value to 0.5)
    feature_ranges = feature_maxs - feature_mins
    feature_ranges[feature_ranges == 0] = 1  # Handle constant features
    
    # Normalize
    # data_normalized = (data - feature_mins) / feature_ranges
    data_normal = (data - feature_mins) / feature_ranges
    data_normalized = (data_normal * 2) - 1
    
    # Verify normalization
    print(f"\nNormalized data range: min={data_normalized.min():.6f}, max={data_normalized.max():.6f}")
    
    # Check if all values are in [-1, 1]
    if data_normalized.min() >= -1 and data_normalized.max() <= 1:
        print("✓ All values are normalized to [-1, 1] range")
    else:
        print("⚠ Warning: Some values are outside [-1, 1] range")
        print(f"   Values below -1: {(data_normalized < -1).sum()}")
        print(f"   Values above 1: {(data_normalized > 1).sum()}")
    
    # Save normalized data
    print(f"\nSaving normalized data to {output_file}...")
    with open(output_file, 'w') as f:
        for i in range(len(data_normalized)):
            # Write normalized features
            normalized_features = [f"{val:.6f}" for val in data_normalized[i]]
            # Write label (keep as is, it's already 0 or 1)
            label = f"{labels[i]:.6f}"
            # Combine and write
            line = ','.join(normalized_features) + ',' + label + '\n'
            f.write(line)
    
    print(f"✓ Saved {len(data_normalized)} normalized samples")
    
    # Save normalization parameters for future use
    np.savez('normalization_params.npz', 
             feature_mins=feature_mins, 
             feature_maxs=feature_maxs,
             feature_ranges=feature_ranges)
    print("✓ Saved normalization parameters to normalization_params.npz")
    
    return data_normalized, labels

if __name__ == "__main__":
    input_file = "roa_labels.txt"
    output_file = "roa_labels_normalized.txt"
    
    normalize_data(input_file, output_file)