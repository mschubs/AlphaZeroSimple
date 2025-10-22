import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TicTacToeModel(nn.Module):
    def __init__(self, board_size, action_size, device):
        super(TicTacToeModel, self).__init__()
        
        self.device = device
        self.board_size = board_size  # 9 for 3x3
        self.action_size = action_size  # 9 for 3x3
        
        # Larger network for more complex game  
        self.fc1 = nn.Linear(in_features=self.board_size, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        
        # Two heads: policy and value
        self.action_head = nn.Linear(in_features=32, out_features=self.action_size)
        self.value_head = nn.Linear(in_features=32, out_features=1)
        
        self.to(device)
    
    def forward(self, x):
        # Flatten input if needed
        x = x.view(-1, self.board_size)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        action_logits = self.action_head(x)
        value_logit = self.value_head(x)
        
        return F.softmax(action_logits, dim=1), torch.tanh(value_logit)
    
    def predict(self, board):
        """Predict policy and value for a board state"""
        board_flat = board.flatten().astype(np.float32)
        board_tensor = torch.FloatTensor(board_flat).to(self.device)
        board_tensor = board_tensor.view(1, self.board_size)
        
        self.eval()
        with torch.no_grad():
            pi, v = self.forward(board_tensor)
        
        return pi.data.cpu().numpy()[0], v.data.cpu().numpy()[0] 