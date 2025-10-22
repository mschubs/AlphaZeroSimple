#!/usr/bin/env python3

import torch
import numpy as np
from connect4_game import Connect4Game
from connect4_model import Connect4Model
from monte_carlo_tree_search import MCTS

def play_connect4():
    """Play Connect 4 against the AI"""
    
    # Setup
    device = torch.device('cpu')
    game = Connect4Game()
    
    # Try to load trained model
    try:
        model = Connect4Model(game.get_board_size(), game.get_action_size(), device)
        checkpoint = torch.load('connect4_current.pth', map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        print("Loaded trained Connect 4 model!")
    except FileNotFoundError:
        print("No trained model found. Train first with: python connect4_main.py")
        return
    
    # MCTS args
    args = {
        'num_simulations': 500,
        'batch_size': 64,
        'numIters': 50,
        'numEps': 25,
        'numItersForTrainExamplesHistory': 20,
        'epochs': 6,
        'checkpoint_path': 'connect4_current.pth'
    }
    
    print("=== Connect 4 vs AI ===")
    print("You are X (player 1), AI is O (player -1)")
    print("Enter column number (0-6) to drop your piece")
    print("Columns: 0 1 2 3 4 5 6")
    print()
    
    # Game loop
    state = game.get_init_board()
    current_player = 1  # Human starts
    
    while True:
        game.print_board(state)
        
        if current_player == 1:

            # AI turn
            print("AI is thinking...")
            
            canonical_board = game.get_canonical_board(state, current_player)
            mcts = MCTS(game, model, args)
            root = mcts.run(model, canonical_board, to_play=1)
            
            # Get move probabilities
            action_probs = [0 for _ in range(game.get_action_size())]
            for k, v in root.children.items():
                action_probs[k] = v.visit_count
            
            if sum(action_probs) > 0:
                action_probs = np.array(action_probs) / sum(action_probs)
                move = root.select_action(temperature=0.1)  # Low temperature for strong play
                
                print(f"AI chooses column {move}")
                print(f"AI's move probabilities: {[f'{p:.3f}' for p in action_probs]}")
                
                state, current_player = game.get_next_state(state, current_player, move)
            else:
                print("AI has no valid moves (this shouldn't happen)")
                break
            
        else:
            # Human turn
            try:
                valid_moves = game.get_valid_moves(state)
                print(f"Valid moves: {[i for i, v in enumerate(valid_moves) if v == 1]}")
                
                move = int(input("Your move (column 0-6): "))
                
                if move < 0 or move >= game.get_action_size():
                    print("Invalid column! Choose 0-6")
                    continue
                
                if valid_moves[move] == 0:
                    print("Column full! Choose another column")
                    continue
                
                state, current_player = game.get_next_state(state, current_player, move)
                
            except (ValueError, KeyboardInterrupt):
                print("\nGame ended by user")
                break
        
        # Check for game end
        reward = game.get_reward_for_player(state, current_player)
        if reward is not None:
            game.print_board(state)
            
            if reward == 1:
                winner = "You" if current_player == -1 else "AI"
                print(f"🎉 {winner} wins!")
            elif reward == -1:
                winner = "AI" if current_player == -1 else "You"
                print(f"🎉 {winner} wins!")
            else:
                print("🤝 It's a draw!")
            
            break
        
        if not game.has_legal_moves(state):
            print("🤝 It's a draw!")
            break

if __name__ == "__main__":
    play_connect4() 