import torch
import multiprocessing as mp

from connect4_game import Connect4Game
from connect4_model import Connect4Model
from connect4_trainer import Connect4Trainer
from connect4_trainer_parallel import Connect4TrainerParallel

def main():
    # Use CPU for now to avoid MPS numerical issues
    device = torch.device('cpu')
    print("Using CPU (avoiding MPS numerical instability)")

    args = {
        'batch_size': 128,
        'numIters': 50,                                # More iterations for complex game
        'num_simulations': 500,                        # More simulations for better play
        'numEps': 1000,                                  # Fewer episodes per iteration (longer games)
        'numItersForTrainExamplesHistory': 20,
        'epochs': 2,                                   # More epochs for complex game
        'checkpoint_path': 'connect4_current.pth',      # Checkpoint file
        'num_evaluations': 200,
        'num_workers': min(mp.cpu_count(), 8),         # Cap workers to avoid overload
        
        # Dirichlet noise parameters for exploration
        'dirichlet_alpha': 0.3,                       # 0.3 for Connect4 (vs 0.03 for Go)
        'dirichlet_epsilon': 0.25,                    # Standard AlphaZero value
    }

    game = Connect4Game()
    board_size = game.get_board_size()
    action_size = game.get_action_size()

    print(f"Connect 4 - Board size: {board_size}, Action size: {action_size}")
    print(f"Using {args['num_workers']} parallel workers")

    model = Connect4Model(board_size, action_size, device)

    # Use parallel trainer for better performance
    trainer = Connect4TrainerParallel(game, model, args)
    trainer.learn()

if __name__ == '__main__':
    # Required for multiprocessing on macOS/Windows
    mp.set_start_method('spawn', force=True)
    main() 