import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from pathlib import Path

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class HumanoidDataset(Dataset):
    """Dataset class for humanoid joint data"""
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class HumanoidCNN(nn.Module):
    """1D CNN for humanoid joint classification"""
    def __init__(self, num_features=67, num_classes=2):
        super(HumanoidCNN, self).__init__()
        
        # Reshape input: (batch, 67) -> (batch, 1, 67) for Conv1d
        # We treat the 67 joints as a sequence
        
        # First convolutional block
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        # Second convolutional block
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # Third convolutional block
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=7, padding=3)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        
        # Calculate flattened size after convolutions
        # Input: 67 -> after pool1: 33 -> after pool2: 16 -> after pool3: 8
        self.flattened_size = 256 * 8
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.relu4 = nn.ReLU()
        
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.relu5 = nn.ReLU()
        
        self.fc3 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        # Reshape: (batch, 67) -> (batch, 1, 67)
        x = x.unsqueeze(1)
        
        # Convolutional blocks
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.relu4(x)
        
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.relu5(x)
        
        x = self.fc3(x)
        
        return x

def load_data(data_file):
    """Load normalized data"""
    print(f"Loading data from {data_file}...")
    data = np.loadtxt(data_file, delimiter=',')
    
    # Separate features and labels
    features = data[:, :-1]  # First 67 columns
    labels = data[:, -1].astype(int)  # Last column (0 or 1)
    
    print(f"Loaded {len(features)} samples")
    print(f"Features shape: {features.shape}")
    print(f"Label distribution: {np.sum(labels == 0)} zeros, {np.sum(labels == 1)} ones")
    print(f"Success rate: {np.mean(labels == 1):.2%}")
    
    return features, labels

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for features, labels in dataloader:
        features, labels = features.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    return epoch_loss, accuracy, precision, recall, f1, all_preds, all_labels

def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Plot losses
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training history saved to {save_path}")

def main():
    # Configuration
    data_file = "roa_labels_normalized.txt"
    batch_size = 64
    num_epochs = 50
    learning_rate = 0.001
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    features, labels = load_data(data_file)
    
    # Create dataset
    dataset = HumanoidDataset(features, labels)
    
    # Split dataset
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    print(f"\nDataset split:")
    print(f"  Training: {len(train_dataset)} samples")
    print(f"  Validation: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = HumanoidCNN(num_features=67, num_classes=2).to(device)
    print(f"\nModel created:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    best_model_state = None
    
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    # Training loop
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_prec, val_rec, val_f1, _, _ = validate(
            model, val_loader, criterion, device
        )
        
        # Save history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc * 100)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
            print(f"  Val Precision: {val_prec:.4f}, Recall: {val_rec:.4f}, F1: {val_f1:.4f}")
            print()
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model (validation accuracy: {best_val_acc*100:.2f}%)")
    
    # Test evaluation
    print("\n" + "="*60)
    print("Evaluating on test set...")
    print("="*60)
    
    test_loss, test_acc, test_prec, test_rec, test_f1, test_preds, test_labels = validate(
        model, test_loader, criterion, device
    )
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Test Precision: {test_prec:.4f}")
    print(f"Test Recall: {test_rec:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=['Failure', 'Success']))
    
    # Save model
    model_path = "humanoid_cnn_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {'num_features': 67, 'num_classes': 2},
    }, model_path)
    print(f"\nModel saved to {model_path}")
    
    # Plot training history
    plot_training_history(train_losses, val_losses, train_accs, val_accs, "training_history.png")
    
    print("\nTraining completed!")

if __name__ == "__main__":
    main()

