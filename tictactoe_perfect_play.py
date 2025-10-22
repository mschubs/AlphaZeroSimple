import numpy as np
from tictactoe_game import TicTacToeGame
from tictactoe_model import TicTacToeModel
import torch


def load_model(checkpoint_path='tictactoe_latest.pth'):
    """Load the trained model from checkpoint"""
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


def minimax(board, player, game, depth=0, max_depth=15):
    """
    Minimax algorithm to find perfect play for TicTacToe
    Returns (best_score, best_action)
    """
    
    # Check if game is over
    reward = game.get_reward_for_player(board, player)
    if reward is not None:
        return reward, None
    
    # Prevent infinite recursion
    if depth >= max_depth:
        return 0, None
    
    valid_moves = game.get_valid_moves(board)
    valid_actions = [i for i, valid in enumerate(valid_moves) if valid]
    
    if not valid_actions:
        return 0, None
    
    if player == 1:  # Maximizing player
        best_score = -float('inf')
        best_action = valid_actions[0]
        
        for action in valid_actions:
            new_board, new_player = game.get_next_state(board, player, action)
            score, _ = minimax(new_board, new_player, game, depth + 1, max_depth)
            
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_score, best_action
    
    else:  # Minimizing player
        best_score = float('inf')
        best_action = valid_actions[0]
        
        for action in valid_actions:
            new_board, new_player = game.get_next_state(board, player, action)
            score, _ = minimax(new_board, new_player, game, depth + 1, max_depth)
            
            if score < best_score:
                best_score = score
                best_action = action
        
        return best_score, best_action


def test_ai_against_perfect_play(model, game, num_games=20):
    """Test AI against perfect minimax play"""
    from monte_carlo_tree_search import MCTS
    
    args = {'num_simulations': 200}
    wins = {'ai': 0, 'perfect': 0, 'draws': 0}
    
    for game_num in range(num_games):
        board = game.get_init_board()
        current_player = 1
        
        # Alternate who goes first
        ai_goes_first = (game_num % 2 == 0)
        
        print(f"Game {game_num + 1}: {'AI' if ai_goes_first else 'Perfect'} goes first")
        
        while True:
            should_ai_play = (current_player == 1 and ai_goes_first) or (current_player == -1 and not ai_goes_first)
            
            if should_ai_play:  # AI turn
                mcts = MCTS(game, model, args)
                canonical_board = game.get_canonical_board(board, current_player)
                root = mcts.run(model, canonical_board, to_play=1)
                action = root.select_action(temperature=0)
                print(f"  AI plays position {action}")
            else:  # Perfect play turn
                canonical_board = game.get_canonical_board(board, current_player)
                _, action = minimax(canonical_board, 1, game)  # Always think as player 1
                print(f"  Perfect plays position {action}")
            
            board, current_player = game.get_next_state(board, current_player, action)
            
            # Check game result from AI's perspective
            if ai_goes_first:
                ai_reward = game.get_reward_for_player(board, 1)
            else:
                ai_reward = game.get_reward_for_player(board, -1)
            
            if ai_reward is not None:
                if ai_reward == 1:
                    wins['ai'] += 1
                    print(f"  AI wins!")
                elif ai_reward == -1:
                    wins['perfect'] += 1
                    print(f"  Perfect play wins!")
                else:
                    wins['draws'] += 1
                    print(f"  Draw!")
                break
        print()
    
    print(f"\nResults after {num_games} games:")
    print(f"AI wins: {wins['ai']}")
    print(f"Perfect play wins: {wins['perfect']}")
    print(f"Draws: {wins['draws']}")
    
    win_rate = wins['ai'] / num_games
    draw_rate = wins['draws'] / num_games
    print(f"AI win rate: {win_rate:.1%}")
    print(f"Draw rate: {draw_rate:.1%}")
    
    return wins


def analyze_opening_moves(game):
    """Analyze the value of different opening moves in TicTacToe"""
    print("TICTACTOE OPENING ANALYSIS")
    print("=" * 50)
    
    initial_board = game.get_init_board()
    
    for action in range(9):
        board, _ = game.get_next_state(initial_board, 1, action)
        score, best_response = minimax(board, -1, game)
        
        # Convert action to board position
        row, col = action // 3, action % 3
        
        print(f"Opening move ({row},{col}) - position {action}: score = {-score}")
        if best_response is not None:
            resp_row, resp_col = best_response // 3, best_response % 3
            print(f"  Best response: ({resp_row},{resp_col}) - position {best_response}")
    
    print("\nNote: Score from first player's perspective")
    print("Score 1 = first player wins, 0 = draw, -1 = second player wins")


if __name__ == "__main__":
    game = TicTacToeGame()
    
    print("TicTacToe Perfect Play Analysis")
    print("=" * 50)
    
    # First analyze opening moves
    analyze_opening_moves(game)
    
    # Then test against trained model if available
    try:
        model, _ = load_model()
        print("\n" + "=" * 50)
        print("Testing trained model against perfect play...")
        test_ai_against_perfect_play(model, game)
    except FileNotFoundError:
        print("\nNo trained model found. Train a model first using tictactoe_main.py") 