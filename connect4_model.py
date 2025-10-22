import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Connect4Model(nn.Module):
    def __init__(self, board_size, action_size, device):
        super(Connect4Model, self).__init__()
        
        self.device = device
        self.board_size = board_size  # 42 for 6x7
        self.action_size = action_size  # 7 for columns
        self.height = 6
        self.width = 7
        
        # Convolutional layers for spatial feature extraction
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Input: 1x6x7, Output: 32x6x7
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Output: 64x6x7
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Output: 128x6x7
        
        # Batch normalization for stable training
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Residual block
        self.conv_res1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv_res2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn_res1 = nn.BatchNorm2d(128)
        self.bn_res2 = nn.BatchNorm2d(128)
        
        # Calculate the size after convolutions (128 channels * 6 * 7)
        self.conv_output_size = 128 * 6 * 7
        
        # Fully connected layers for final processing
        self.fc1 = nn.Linear(self.conv_output_size, 256)
        self.fc2 = nn.Linear(256, 128)
        
        # Two heads: policy and value
        self.action_head = nn.Linear(128, self.action_size)  # 7 actions (columns)
        self.value_head = nn.Linear(128, 1)  # 1 value
        
        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.2)
        
        self.to(device)
    
    def forward(self, x):
        # Reshape input to (batch_size, 1, 6, 7) for conv layers
        x = x.view(-1, 1, self.height, self.width)
        
        # Convolutional layers with ReLU and batch norm
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Residual block
        residual = x
        x = F.relu(self.bn_res1(self.conv_res1(x)))
        x = self.bn_res2(self.conv_res2(x))
        x = F.relu(x + residual)  # Skip connection
        
        # Flatten for fully connected layers
        x = x.view(-1, self.conv_output_size)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Output heads
        action_logits = self.action_head(x)
        value_logit = self.value_head(x)
        
        return F.softmax(action_logits, dim=1), torch.tanh(value_logit)
    
    def predict(self, board):
        """Predict policy and value for a board state"""
        # Convert board to tensor and add batch dimension
        board_tensor = torch.FloatTensor(board).to(self.device)
        board_tensor = board_tensor.view(1, 1, self.height, self.width)
        
        self.eval()
        with torch.no_grad():
            pi, v = self.forward(board_tensor)
        
        return pi.data.cpu().numpy()[0], v.data.cpu().numpy()[0] 