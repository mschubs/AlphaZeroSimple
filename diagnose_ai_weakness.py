#!/usr/bin/env python3

import torch
import numpy as np
from connect4_game import Connect4Game
from connect4_model import Connect4Model
from monte_carlo_tree_search import MCTS

def create_winning_positions():
    """Create test positions where there's an obvious winning move"""
    game = Connect4Game()
    positions = []
    
    # Position 1: Horizontal win opportunity
    board1 = game.get_init_board()
    board1[5][0] = 1  # X X X _ _ _ _
    board1[5][1] = 1
    board1[5][2] = 1
    # Winning move: column 3
    positions.append(("Horizontal Win", board1, 3))
    
    # Position 2: Vertical win opportunity  
    board2 = game.get_init_board()
    board2[5][3] = 1  # Column 3: X X X _ _ _
    board2[4][3] = 1
    board2[3][3] = 1
    # Winning move: column 3
    positions.append(("Vertical Win col 3", board2, 3))

     # Position 2: Vertical win opportunity  
    board2 = game.get_init_board()
    board2[5][4] = 1  # Column 3: X X X _ _ _
    board2[4][4] = 1
    board2[3][4] = 1
    # Winning move: column 3
    positions.append(("Vertical Win col 4", board2, 4))
    
    # Position 3: Diagonal win opportunity
    board3 = game.get_init_board()
    board3[4][2] = 1  # Diagonal: X _ _ _ _ _ _
    board3[3][3] = 1  #          _ X _ _ _ _ _
    board3[2][4] = 1  #          _ _ X _ _ _ _
    board3[5][0] = -1  # Add some opponent pieces
    board3[4][0] = -1
    board3[3][0] = -1
    board3[5][2] = -1  # Diagonal: X _ _ _ _ _ _
    board3[4][3] = -1  #          _ X _ _ _ _ _
    board3[3][4] = -1
    board3[5][3] = 1  #          _ X _ _ _ _ _
    board3[4][4] = -1
    board3[5][4] = -1
    # Winning move: column 4 (completes diagonal)
    positions.append(("Diagonal Win", board3, 1))
    
    # Position 4: Blocking opponent win
    board4 = game.get_init_board()
    board4[5][2] = -1  # Opponent has O O O _ _ _ _
    board4[5][3] = -1
    board4[5][4] = -1
    board4[5][5] = 1
    # Must block at column 1 or 5
    positions.append(("Block Opponent", board4, 1))
    
    return positions

def analyze_ai_decision(game, model, board, expected_move, position_name):
    """Analyze why AI made its decision"""
    print(f"\n=== ANALYZING: {position_name} ===")
    
    # Show the position
    print("Board position:")
    game.print_board(board)
    print(f"Expected winning move: Column {expected_move}")
    
    # Get raw neural network prediction
    pi_raw, v_raw = model.predict(board)
    print(f"\nRaw neural network output:")
    print(f"Policy: {[f'{p:.3f}' for p in pi_raw]}")
    print(f"Value: {v_raw[0]:.3f}")
    print(f"NN prefers column: {np.argmax(pi_raw)}")
    print(f"NN confidence in expected move: {pi_raw[expected_move]:.3f}")
    
    # Get MCTS decision
    args = {'num_simulations': 100}
    mcts = MCTS(game, model, args)
    
    canonical_board = game.get_canonical_board(board, 1)  # Player 1's turn
    root = mcts.run(model, canonical_board, to_play=1)
    
    # Get MCTS visit counts
    action_probs = [0 for _ in range(game.get_action_size())]
    for k, v in root.children.items():
        action_probs[k] = v.visit_count
    
    if sum(action_probs) > 0:
        action_probs = np.array(action_probs) / sum(action_probs)
        mcts_choice = np.argmax(action_probs)
        
        print(f"\nMCTS results:")
        print(f"Visit counts: {[int(v.visit_count) if k in root.children else 0 for k in range(game.get_action_size())]}")
        print(f"Probabilities: {[f'{p:.3f}' for p in action_probs]}")
        print(f"MCTS prefers column: {mcts_choice}")
        print(f"MCTS confidence in expected move: {action_probs[expected_move]:.3f}")
        
        # Check if AI found the winning move
        if mcts_choice == expected_move:
            print("✅ AI FOUND the winning move!")
        else:
            print("❌ AI MISSED the winning move!")
            print(f"AI chose column {mcts_choice} instead of {expected_move}")
    
    # Verify the expected move actually wins
    test_board, _ = game.get_next_state(board, 1, expected_move)
    if game.is_win(test_board, 1):
        print(f"✅ Confirmed: Column {expected_move} is indeed a winning move")
    else:
        print(f"❓ Note: Column {expected_move} may not be an immediate win")
    
    return mcts_choice == expected_move

def diagnose_ai_problems():
    """Diagnose common AI problems"""
    print("=== DIAGNOSING AI DECISION MAKING ===\n")
    
    # Setup
    device = torch.device('cpu')
    game = Connect4Game()
    
    # Try to load model
    try:
        model = Connect4Model(game.get_board_size(), game.get_action_size(), device)
        checkpoint = torch.load('connect4_latest.pth', map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        print("✅ Loaded Connect 4 model")
    except FileNotFoundError:
        print("❌ No trained model found. Please train first.")
        return
    
    # Test on winning positions
    positions = create_winning_positions()
    results = []
    
    for position_name, board, expected_move in positions:
        found_win = analyze_ai_decision(game, model, board, expected_move, position_name)
        results.append((position_name, found_win))
    
    # Summary
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)
    
    wins_found = sum(1 for _, found in results if found)
    total_positions = len(results)
    
    print(f"AI found {wins_found}/{total_positions} winning moves")
    
    for position_name, found in results:
        status = "✅ FOUND" if found else "❌ MISSED"
        print(f"{status}: {position_name}")
    
    if wins_found < total_positions:
        print(f"\n🔍 POSSIBLE REASONS FOR MISSING WINS:")
        print("1. 🧠 Neural Network Issues:")
        print("   - Network not fully trained")
        print("   - Poor value/policy estimates")
        print("   - Insufficient training data")
        
        print("\n2. 🌳 MCTS Issues:")
        print("   - Too few simulations")
        print("   - Poor exploration/exploitation balance")
        print("   - Temperature too high (adds randomness)")
        
        print("\n3. 📚 Training Data Issues:")
        print("   - Missing similar winning patterns")
        print("   - Self-play brittleness")
        print("   - Insufficient diverse opponents")
        
        print("\n4. 🔧 SOLUTIONS:")
        print("   - Increase MCTS simulations")
        print("   - Train longer with curriculum learning")
        print("   - Add tactical training positions")
        print("   - Use temperature=0 for best play")
        print("   - Add position evaluation bonuses")

def test_mcts_simulations():
    """Test how MCTS performance changes with simulation count"""
    print("\n=== TESTING MCTS SIMULATION COUNT ===")
    
    device = torch.device('cpu')
    game = Connect4Game()
    
    try:
        model = Connect4Model(game.get_board_size(), game.get_action_size(), device)
        checkpoint = torch.load('connect4_latest.pth', map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
    except FileNotFoundError:
        print("❌ No trained model found.")
        return
    
    # Simple winning position
    board = game.get_init_board()
    board[5][0] = 1
    board[5][1] = 1
    board[5][2] = 1
    # Winning move: column 3
    
    print("Test position (horizontal win opportunity):")
    game.print_board(board)
    print("Correct move: Column 3")
    
    simulation_counts = [10, 50, 100, 200, 500]
    
    for sim_count in simulation_counts:
        args = {'num_simulations': sim_count}
        mcts = MCTS(game, model, args)
        
        canonical_board = game.get_canonical_board(board, 1)
        root = mcts.run(model, canonical_board, to_play=1)
        
        # Get most visited action
        if root.children:
            best_action = max(root.children.keys(), key=lambda k: root.children[k].visit_count)
            best_visits = root.children[best_action].visit_count
            correct_visits = root.children.get(3, None)
            correct_visits = correct_visits.visit_count if correct_visits else 0
            
            status = "✅" if best_action == 3 else "❌"
            print(f"{status} {sim_count:3d} sims: Best=Col{best_action}({best_visits}), Correct=Col3({correct_visits})")

if __name__ == "__main__":
    diagnose_ai_problems()
    test_mcts_simulations() 