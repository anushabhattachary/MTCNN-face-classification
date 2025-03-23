import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import os
from cnn_model import GenderEthnicityModel

# Set device (use GPU if available)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():  # Check for Apple Silicon GPU
    device = torch.device("mps")
print(f"Using device: {device}")

# Load dataset
train_images = np.load("datasets/train_images.npy")
gender_labels = np.load("datasets/gender_labels.npy")
ethnicity_labels = np.load("datasets/ethnicity_labels.npy")

# Convert numpy arrays to PyTorch tensors
train_images = torch.from_numpy(train_images).permute(0, 3, 1, 2).float()  # Change from (N,H,W,C) to (N,C,H,W)
gender_labels = torch.from_numpy(gender_labels).long()
ethnicity_labels = torch.from_numpy(ethnicity_labels).long()

# Create dataset and dataloader
dataset = TensorDataset(train_images, gender_labels, ethnicity_labels)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize model
model = GenderEthnicityModel().to(device)

# Loss functions and optimizer
gender_criterion = nn.CrossEntropyLoss()
ethnicity_criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_gender_loss = 0.0
    running_ethnicity_loss = 0.0
    total_gender_correct = 0
    total_ethnicity_correct = 0
    total = 0
    
    for inputs, gender_targets, ethnicity_targets in train_loader:
        inputs = inputs.to(device)
        gender_targets = gender_targets.to(device)
        ethnicity_targets = ethnicity_targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        gender_outputs, ethnicity_outputs = model(inputs)
        
        # Calculate losses
        gender_loss = gender_criterion(gender_outputs, gender_targets)
        ethnicity_loss = ethnicity_criterion(ethnicity_outputs, ethnicity_targets)
        total_loss = gender_loss + ethnicity_loss
        
        # Backward pass and optimize
        total_loss.backward()
        optimizer.step()
        
        # Statistics
        running_gender_loss += gender_loss.item() * inputs.size(0)
        running_ethnicity_loss += ethnicity_loss.item() * inputs.size(0)
        
        _, gender_preds = torch.max(gender_outputs, 1)
        _, ethnicity_preds = torch.max(ethnicity_outputs, 1)
        
        total_gender_correct += (gender_preds == gender_targets).sum().item()
        total_ethnicity_correct += (ethnicity_preds == ethnicity_targets).sum().item()
        total += inputs.size(0)
    
    # Validation phase
    model.eval()
    val_gender_loss = 0.0
    val_ethnicity_loss = 0.0
    val_gender_correct = 0
    val_ethnicity_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for inputs, gender_targets, ethnicity_targets in val_loader:
            inputs = inputs.to(device)
            gender_targets = gender_targets.to(device)
            ethnicity_targets = ethnicity_targets.to(device)
            
            # Forward pass
            gender_outputs, ethnicity_outputs = model(inputs)
            
            # Calculate losses
            gender_loss = gender_criterion(gender_outputs, gender_targets)
            ethnicity_loss = ethnicity_criterion(ethnicity_outputs, ethnicity_targets)
            
            # Statistics
            val_gender_loss += gender_loss.item() * inputs.size(0)
            val_ethnicity_loss += ethnicity_loss.item() * inputs.size(0)
            
            _, gender_preds = torch.max(gender_outputs, 1)
            _, ethnicity_preds = torch.max(ethnicity_outputs, 1)
            
            val_gender_correct += (gender_preds == gender_targets).sum().item()
            val_ethnicity_correct += (ethnicity_preds == ethnicity_targets).sum().item()
            val_total += inputs.size(0)
    
    # Print statistics
    epoch_gender_loss = running_gender_loss / total
    epoch_ethnicity_loss = running_ethnicity_loss / total
    gender_acc = total_gender_correct / total * 100
    ethnicity_acc = total_ethnicity_correct / total * 100
    
    val_gender_loss = val_gender_loss / val_total
    val_ethnicity_loss = val_ethnicity_loss / val_total
    val_gender_acc = val_gender_correct / val_total * 100
    val_ethnicity_acc = val_ethnicity_correct / val_total * 100
    
    print(f'Epoch {epoch+1}/{num_epochs}:')
    print(f'Train - Gender Loss: {epoch_gender_loss:.4f}, Ethnicity Loss: {epoch_ethnicity_loss:.4f}')
    print(f'Train - Gender Acc: {gender_acc:.2f}%, Ethnicity Acc: {ethnicity_acc:.2f}%')
    print(f'Val - Gender Loss: {val_gender_loss:.4f}, Ethnicity Loss: {val_ethnicity_loss:.4f}')
    print(f'Val - Gender Acc: {val_gender_acc:.2f}%, Ethnicity Acc: {val_ethnicity_acc:.2f}%')
    print('-' * 60)

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save the model
torch.save(model.state_dict(), "models/gender_ethnicity_model.pth")
print("Model saved successfully!")