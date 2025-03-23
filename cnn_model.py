import torch
import torch.nn as nn
import torch.nn.functional as F

class GenderEthnicityModel(nn.Module):
    def __init__(self):
        super(GenderEthnicityModel, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate input size for the first fully connected layer
        # After 3 pooling layers with stride 2, the size is reduced by 2^3 = 8
        # So 128x128 becomes 16x16
        fc_input_size = 128 * 16 * 16
        
        # Fully connected layers
        self.fc = nn.Linear(fc_input_size, 128)
        self.dropout = nn.Dropout(0.5)
        
        # Output layers
        self.gender_output = nn.Linear(128, 2)
        self.ethnicity_output = nn.Linear(128, 5)
        
    def forward(self, x):
        # Convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layer
        x = F.relu(self.fc(x))
        x = self.dropout(x)
        
        # Output layers
        gender = self.gender_output(x)
        ethnicity = self.ethnicity_output(x)
        
        return gender, ethnicity