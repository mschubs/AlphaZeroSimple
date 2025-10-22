import numpy as np

def get_all_symmetries(board, policy):
    """
    Get all 8 symmetries of a TicTacToe position
    Returns list of (board, policy) pairs
    """
    symmetries = []
    
    # Original
    symmetries.append((board.copy(), policy.copy()))
    
    # 90 degree rotations
    for i in range(1, 4):
        rotated_board = np.rot90(board, i)
        rotated_policy = rotate_policy_90(policy, i)
        symmetries.append((rotated_board, rotated_policy))
    
    # Horizontal flip + rotations
    flipped_board = np.fliplr(board)
    flipped_policy = flip_policy_horizontal(policy)
    symmetries.append((flipped_board, flipped_policy))
    
    for i in range(1, 4):
        rotated_board = np.rot90(flipped_board, i)
        rotated_policy = rotate_policy_90(flipped_policy, i)
        symmetries.append((rotated_board, rotated_policy))
    
    return symmetries

def rotate_policy_90(policy, num_rotations):
    """Rotate policy 90 degrees num_rotations times"""
    # Convert 1D policy to 3x3 grid
    policy_grid = policy.reshape(3, 3)
    
    # Rotate the grid
    rotated_grid = np.rot90(policy_grid, num_rotations)
    
    # Convert back to 1D
    return rotated_grid.flatten()

def flip_policy_horizontal(policy):
    """Flip policy horizontally"""
    # Convert 1D policy to 3x3 grid
    policy_grid = policy.reshape(3, 3)
    
    # Flip horizontally
    flipped_grid = np.fliplr(policy_grid)
    
    # Convert back to 1D
    return flipped_grid.flatten()

def augment_training_data(examples, game=None):
    """
    Augment training data with all symmetries
    Input: list of (board, policy, value) tuples
    Output: expanded list with all symmetries
    """
    augmented = []
    
    for board, policy, value in examples:
        if game and hasattr(game, 'get_symmetries'):
            # Use game-specific symmetries
            symmetries = game.get_symmetries(board, policy)
        else:
            # Fall back to TicTacToe symmetries for backward compatibility
            symmetries = get_all_symmetries(board, policy)
        
        # Add all symmetries to training data
        for sym_board, sym_policy in symmetries:
            augmented.append((sym_board, sym_policy, value))
    
    return augmented

# Test the symmetry functions
if __name__ == "__main__":
    # Test with a simple board
    board = np.array([[1, 0, 0],
                      [0, -1, 0], 
                      [0, 0, 0]])
    
    # Test policy (prefer center-right)
    policy = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.4, 0.05, 0.05, 0.05])
    
    print("Original board:")
    print(board)
    print("Original policy:", policy)
    
    symmetries = get_all_symmetries(board, policy)
    
    print(f"\nGenerated {len(symmetries)} symmetries:")
    for i, (sym_board, sym_policy) in enumerate(symmetries):
        print(f"\nSymmetry {i}:")
        print(sym_board)
        print("Policy:", [f"{p:.2f}" for p in sym_policy]) 