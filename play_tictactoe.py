import torch
import numpy as np
from tictactoe_game import TicTacToeGame
from tictactoe_model import TicTacToeModel
from monte_carlo_tree_search import MCTS

def load_model(checkpoint_path='tictactoe_latest_test1.pth'):
    """Load the trained TicTacToe model from checkpoint"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    game = TicTacToeGame()
    board_size = game.get_board_size()
    action_size = game.get_action_size()
    
    model = TicTacToeModel(board_size, action_size, device)
    
    # Load the trained weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    return model, game

def print_board(board):
    """Print the TicTacToe board in a readable format"""
    symbols = {0: '.', 1: 'X', -1: 'O'}
    print("Current board:")
    for i in range(3):
        row = ' '.join([symbols[int(board[i][j])] for j in range(3)])
        print(f"  {row}")
    print("Position numbers:")
    for i in range(3):
        row = ' '.join([str(i*3 + j) for j in range(3)])
        print(f"  {row}")
    print()

def get_human_move(board, game):
    """Get a move from human player"""
    valid_moves = game.get_valid_moves(board)
    
    while True:
        try:
            move = int(input("Enter your move (0-8): "))
            if 0 <= move <= 8 and valid_moves[move] == 1:
                return move
            else:
                print("Invalid move! Choose an empty position (0-8).")
        except ValueError:
            print("Please enter a number between 0 and 8.")

def get_ai_move(board, model, game, player, num_simulations=200):
    """Get a move from AI using MCTS"""
    args = {'num_simulations': num_simulations}
    mcts = MCTS(game, model, args)
    
    canonical_board = game.get_canonical_board(board, player)
    root = mcts.run(model, canonical_board, to_play=1)
    
    # Get move probabilities
    action_probs = [0 for _ in range(game.get_action_size())]
    for action, child in root.children.items():
        action_probs[action] = child.visit_count
    
    if sum(action_probs) > 0:
        action_probs = np.array(action_probs) / sum(action_probs)
        print(f"AI move probabilities: {[f'{p:.3f}' for p in action_probs]}")
    
    return root.select_action(temperature=0)

def play_human_vs_ai():
    """Play a game: Human vs AI"""
    model, game = load_model()
    board = game.get_init_board()
    current_player = 1  # Human starts as player 1 (X)
    
    print("Welcome to TicTacToe!")
    print("You are X, AI is O")
    print("Goal: Get 3 pieces in a row (horizontal, vertical, or diagonal)")
    print()
    
    while True:
        print_board(board)
        
        if current_player == 1:  # Human turn
            print("AI is thinking...")
            action = get_ai_move(board, model, game, current_player)
            print(f"AI plays position {action}")
            
        else:  # AI turn

            print("Your turn!")
            action = get_human_move(board, game)
            
        
        # Make the move
        board, current_player = game.get_next_state(board, current_player, action)
        
        # Check if game is over
        reward = game.get_reward_for_player(board, 1)  # Check for player 1 (human)
        
        if reward is not None:
            print_board(board)
            if reward == 1:
                print("You win! 🎉")
            elif reward == -1:
                print("AI wins! 🤖")
            else:
                print("It's a draw! 🤝")
            break
        
        print()

def play_ai_vs_ai():
    """Watch AI play against itself"""
    model, game = load_model()
    board = game.get_init_board()
    current_player = 1
    
    print("Watching AI vs AI...")
    print("Player 1: X, Player 2: O")
    print()
    
    move_count = 0
    while True:
        print_board(board)
        print(f"Move {move_count + 1}: Player {1 if current_player == 1 else 2}'s turn")
        
        # Get AI move (for current player)
        action = get_ai_move(board, model, game, current_player)
        print(f"Player {1 if current_player == 1 else 2} plays position {action}")
        
        # Make the move
        board, current_player = game.get_next_state(board, current_player, action)
        
        # Check if game is over
        reward = game.get_reward_for_player(board, 1)  # Check from player 1's perspective
        
        if reward is not None:
            print_board(board)
            if reward == 1:
                print("Player 1 (X) wins!")
            elif reward == -1:
                print("Player 2 (O) wins!")
            else:
                print("It's a draw!")
            break
        
        move_count += 1
        print()

def main():
    print("TicTacToe Game with Trained AI")
    print("1. Play against AI")
    print("2. Watch AI vs AI")
    
    while True:
        try:
            choice = int(input("Choose option (1 or 2): "))
            if choice == 1:
                play_human_vs_ai()
                break
            elif choice == 2:
                play_ai_vs_ai()
                break
            else:
                print("Please choose 1 or 2")
        except ValueError:
            print("Please enter a number")

if __name__ == "__main__":
    main() 