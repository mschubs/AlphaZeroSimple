#!/usr/bin/env python3

import torch
import numpy as np
from connect4_game import Connect4Game
from connect4_model import Connect4Model
import torch.optim as optim

def generate_vertical_training_data():
    """Generate training data focused on vertical patterns"""
    game = Connect4Game()
    training_data = []
    
    print("Generating vertical training data...")
    
    # Generate 200 vertical win positions
    for _ in range(200):
        for col in range(7):
            board = game.get_init_board()
            
            # Place 3 pieces vertically with some randomness
            start_row = np.random.randint(2, 6)  # Start from row 2-5
            for i in range(3):
                if start_row - i >= 0:
                    board[start_row - i][col] = 1
            
            # Add some random opponent pieces (not blocking)
            for _ in range(np.random.randint(0, 4)):
                rand_col = np.random.randint(0, 7)
                if rand_col != col:  # Don't block the winning move
                    valid_moves = game.get_valid_moves(board)
                    if valid_moves[rand_col]:
                        board, _ = game.get_next_state(board, -1, rand_col)
            
            # Create strong policy favoring the winning move
            policy = np.ones(7) * 0.01  # Small probability for others
            policy[col] = 0.93  # Very high for winning move
            policy = policy / np.sum(policy)  # Normalize
            
            training_data.append((board.copy(), policy.copy(), 0.95))  # High value
    
    return training_data

def train_on_vertical_patterns():
    """Train model specifically on vertical patterns"""
    device = torch.device('cpu')
    game = Connect4Game()
    
    # Load existing model
    model = Connect4Model(game.get_board_size(), game.get_action_size(), device)
    try:
        checkpoint = torch.load('connect4_latest.pth', map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        print("✅ Loaded existing model")
    except FileNotFoundError:
        print("❌ No existing model found. Train base model first.")
        return
    
    # Generate training data
    training_data = generate_vertical_training_data()
    
    # Train
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(10):
        total_loss = 0
        for board, policy, value in training_data:
            board_tensor = torch.FloatTensor(board).unsqueeze(0).to(device)
            policy_tensor = torch.FloatTensor(policy).unsqueeze(0).to(device)
            value_tensor = torch.FloatTensor([value]).to(device)
            
            # Forward pass
            pred_policy, pred_value = model(board_tensor)
            
            # Compute losses
            policy_loss = -(policy_tensor * torch.log(pred_policy + 1e-8)).sum()
            value_loss = ((value_tensor - pred_value.view(-1))**2).mean()
            loss = policy_loss + value_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}: Average loss = {total_loss/len(training_data):.4f}")
    
    # Save improved model
    torch.save({'state_dict': model.state_dict()}, 'connect4_vertical_trained.pth')
    print("✅ Saved vertically-trained model")

if __name__ == "__main__":
    train_on_vertical_patterns()
