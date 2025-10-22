import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Residual block for deeper chess network"""
    
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class ChessModel(nn.Module):
    """
    Deep convolutional neural network for chess
    Much deeper than Connect4 model due to game complexity
    """
    
    def __init__(self, board_size, action_size, device, num_channels=256, num_residual_blocks=19):
        super(ChessModel, self).__init__()
        
        self.device = device
        self.board_size = board_size
        self.action_size = action_size
        self.num_channels = num_channels
        
        # Initial convolution (12 input channels for piece types)
        self.initial_conv = nn.Conv2d(12, num_channels, kernel_size=3, padding=1)
        self.initial_bn = nn.BatchNorm2d(num_channels)
        
        # Residual blocks (similar to AlphaZero)
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_residual_blocks)
        ])
        
        # Policy head
        self.policy_conv = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, action_size)
        
        # Value head
        self.value_conv = nn.Conv2d(num_channels, 3, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(3)
        self.value_fc1 = nn.Linear(3 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
        self.to(device)
        
    def forward(self, x):
        # Reshape flattened input to 8x8x12
        batch_size = x.shape[0]
        x = x.view(batch_size, 12, 8, 8)
        
        # Initial convolution
        x = F.relu(self.initial_bn(self.initial_conv(x)))
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(batch_size, -1)
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy, dim=1)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(batch_size, -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value
    
    def predict(self, board):
        """
        Predict action probabilities and value for a single board state
        """
        self.eval()
        with torch.no_grad():
            if isinstance(board, np.ndarray):
                board = torch.FloatTensor(board).unsqueeze(0).to(self.device)
            elif len(board.shape) == 1:
                board = board.unsqueeze(0)
            
            log_pi, v = self.forward(board)
            pi = torch.exp(log_pi).cpu().numpy()[0]
            v = v.cpu().numpy()[0][0]
            
        return pi, v 