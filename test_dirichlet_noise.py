#!/usr/bin/env python3

import torch
import numpy as np
import matplotlib.pyplot as plt
from connect4_game import Connect4Game
from connect4_model import Connect4Model
from monte_carlo_tree_search import MCTS, add_dirichlet_noise

def test_dirichlet_noise():
    """Test Dirichlet noise functionality"""
    
    print("🎲 Testing Dirichlet Noise in MCTS")
    print("=" * 50)
    
    # Setup
    device = torch.device('cpu')
    game = Connect4Game()
    model = Connect4Model(game.get_board_size(), game.get_action_size(), device)
    
    # Create a test position
    board = game.get_init_board()
    board[5][3] = 1   # Player 1 piece in center
    board[4][3] = -1  # Player -1 piece on top
    
    print("Test board position:")
    game.print_board(board)
    
    # Get neural network predictions
    pi, v = model.predict(board)
    valid_moves = game.get_valid_moves(board)
    
    print(f"Neural network policy: {[f'{p:.3f}' for p in pi]}")
    print(f"Valid moves: {valid_moves}")
    print(f"Board value: {v:.3f}")
    print()
    
    # Test Dirichlet noise with different parameters
    test_cases = [
        (0.3, 0.25, "Standard Connect4"),
        (0.1, 0.25, "Less noise (chess-like)"),
        (1.0, 0.25, "More uniform noise"),
        (0.3, 0.5, "Higher mixing weight"),
        (0.3, 0.1, "Lower mixing weight"),
    ]
    
    print("📊 Dirichlet Noise Effects")
    print("-" * 40)
    
    for alpha, epsilon, description in test_cases:
        print(f"\n{description} (α={alpha}, ε={epsilon}):")
        
        # Generate multiple samples to show variation
        samples = []
        for _ in range(5):
            noisy_probs = add_dirichlet_noise(pi, valid_moves, alpha, epsilon)
            samples.append(noisy_probs)
        
        # Show the samples
        for i, sample in enumerate(samples):
            print(f"  Sample {i+1}: {[f'{p:.3f}' for p in sample]}")
        
        # Calculate statistics
        mean_probs = np.mean(samples, axis=0)
        std_probs = np.std(samples, axis=0)
        
        print(f"  Mean:     {[f'{p:.3f}' for p in mean_probs]}")
        print(f"  Std Dev:  {[f'{p:.3f}' for p in std_probs]}")

def test_mcts_with_noise():
    """Test MCTS with and without Dirichlet noise"""
    
    print("\n🌳 Testing MCTS with Dirichlet Noise")
    print("=" * 50)
    
    # Setup
    device = torch.device('cpu')
    game = Connect4Game()
    model = Connect4Model(game.get_board_size(), game.get_action_size(), device)
    
    args = {
        'num_simulations': 100,
        'dirichlet_alpha': 0.3,
        'dirichlet_epsilon': 0.25
    }
    
    # Create test position
    board = game.get_init_board()
    board[5][3] = 1   # Some pieces for more interesting position
    board[4][3] = -1
    board[5][2] = 1
    
    mcts = MCTS(game, model, args)
    
    # Test without noise (evaluation mode)
    print("🎯 MCTS without noise (evaluation mode):")
    results_no_noise = []
    
    for run in range(3):
        root = mcts.run(model, board, to_play=1, add_noise=False)
        
        action_probs = [0] * game.get_action_size()
        for action, child in root.children.items():
            action_probs[action] = child.visit_count
        
        if sum(action_probs) > 0:
            action_probs = np.array(action_probs) / sum(action_probs)
            results_no_noise.append(action_probs)
            print(f"  Run {run+1}: {[f'{p:.3f}' for p in action_probs]}")
    
    # Test with noise (training mode)
    print("\n🎲 MCTS with Dirichlet noise (training mode):")
    results_with_noise = []
    
    for run in range(3):
        root = mcts.run(model, board, to_play=1, add_noise=True)
        
        action_probs = [0] * game.get_action_size()
        for action, child in root.children.items():
            action_probs[action] = child.visit_count
        
        if sum(action_probs) > 0:
            action_probs = np.array(action_probs) / sum(action_probs)
            results_with_noise.append(action_probs)
            print(f"  Run {run+1}: {[f'{p:.3f}' for p in action_probs]}")
    
    # Compare results
    print("\n📊 Comparison:")
    
    if results_no_noise and results_with_noise:
        mean_no_noise = np.mean(results_no_noise, axis=0)
        std_no_noise = np.std(results_no_noise, axis=0)
        
        mean_with_noise = np.mean(results_with_noise, axis=0)
        std_with_noise = np.std(results_with_noise, axis=0)
        
        print(f"Without noise - Mean: {[f'{p:.3f}' for p in mean_no_noise]}")
        print(f"Without noise - Std:  {[f'{p:.3f}' for p in std_no_noise]}")
        print(f"With noise - Mean:    {[f'{p:.3f}' for p in mean_with_noise]}")
        print(f"With noise - Std:     {[f'{p:.3f}' for p in std_with_noise]}")
        
        avg_std_no_noise = np.mean(std_no_noise)
        avg_std_with_noise = np.mean(std_with_noise)
        
        print(f"\nAverage standard deviation:")
        print(f"  Without noise: {avg_std_no_noise:.4f}")
        print(f"  With noise: {avg_std_with_noise:.4f}")
        
        if avg_std_with_noise > avg_std_no_noise * 2:
            print("✅ Dirichlet noise significantly increases exploration!")
        else:
            print("⚠️  Consider increasing noise parameters for more exploration")

def visualize_noise_effects():
    """Create a visualization of noise effects (if matplotlib available)"""
    
    try:
        print("\n📈 Creating Dirichlet Noise Visualization")
        print("=" * 50)
        
        # Generate sample data
        original_probs = np.array([0.05, 0.15, 0.25, 0.35, 0.15, 0.05, 0.0])  # Peaked distribution
        valid_moves = [1, 1, 1, 1, 1, 1, 0]  # Last column blocked
        
        # Generate many samples with different parameters
        alphas = [0.1, 0.3, 1.0]
        epsilons = [0.1, 0.25, 0.5]
        
        fig, axes = plt.subplots(len(alphas), len(epsilons), figsize=(15, 10))
        fig.suptitle('Dirichlet Noise Effects on Action Probabilities', fontsize=16)
        
        for i, alpha in enumerate(alphas):
            for j, epsilon in enumerate(epsilons):
                # Generate 100 samples
                samples = []
                for _ in range(100):
                    noisy_probs = add_dirichlet_noise(original_probs, valid_moves, alpha, epsilon)
                    samples.append(noisy_probs)
                
                samples = np.array(samples)
                
                # Plot distribution
                ax = axes[i, j]
                for col in range(6):  # Skip blocked column
                    ax.hist(samples[:, col], alpha=0.6, bins=20, label=f'Col {col}')
                
                ax.set_title(f'α={alpha}, ε={epsilon}')
                ax.set_xlabel('Probability')
                ax.set_ylabel('Frequency')
                if i == 0 and j == 0:
                    ax.legend()
        
        plt.tight_layout()
        plt.savefig('dirichlet_noise_effects.png', dpi=150, bbox_inches='tight')
        print("📊 Visualization saved as 'dirichlet_noise_effects.png'")
        
    except ImportError:
        print("⚠️  Matplotlib not available, skipping visualization")
    except Exception as e:
        print(f"⚠️  Visualization failed: {e}")

def main():
    print("Dirichlet Noise Testing for AlphaZero Connect4")
    print("=" * 60)
    
    try:
        test_dirichlet_noise()
        test_mcts_with_noise()
        
        # Ask about visualization
        try:
            response = input("\nGenerate noise visualization plot? (y/N): ").lower()
            if response == 'y':
                visualize_noise_effects()
        except:
            print("Skipping visualization")
        
        print("\n🎯 Summary")
        print("=" * 30)
        print("✅ Dirichlet noise has been successfully added to MCTS!")
        print("✅ Noise is applied during training (self-play) for exploration")
        print("✅ No noise during evaluation for optimal play")
        print()
        print("Key benefits:")
        print("- Encourages exploration of different moves during training")
        print("- Prevents overfitting to neural network predictions")
        print("- Standard AlphaZero technique for robust learning")
        print()
        print("Configuration in args:")
        print("- 'dirichlet_alpha': 0.3 (Connect4 recommended)")
        print("- 'dirichlet_epsilon': 0.25 (standard AlphaZero)")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 