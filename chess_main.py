import torch
import multiprocessing as mp

from chess_game import ChessGame
from chess_model import ChessModel
from trainer import Trainer  # Reuse existing trainer

def main():
    # Use CPU for now (chess training is very memory intensive)
    device = torch.device('cpu')
    print("Training AlphaZero for Chess")
    print("Using CPU (chess models are very large)")

    # Chess-specific hyperparameters
    args = {
        'batch_size': 32,                    # Smaller batch due to large model
        'numIters': 100,                     # Chess needs many iterations
        'num_simulations': 800,              # More simulations for chess complexity
        'numEps': 100,                       # Fewer episodes (games are longer)
        'numItersForTrainExamplesHistory': 10,
        'epochs': 3,                         # More epochs for complex patterns
        'checkpoint_path': 'chess_latest.pth',
        'num_evaluations': 40,               # Fewer evaluations (games are slow)
        
        # Dirichlet noise for chess (similar to Go)
        'dirichlet_alpha': 0.03,             # Lower alpha for chess (more focused)
        'dirichlet_epsilon': 0.25,          # Standard AlphaZero value
        
        # Learning rate and regularization
        'lr': 0.001,                         # Lower learning rate for stability
        'weight_decay': 1e-4,                # L2 regularization
        
        # Temperature settings
        'temp_threshold': 15,                # Moves before temperature=0
        'cpuct': 1.0,                        # UCB exploration constant
    }

    game = ChessGame()
    board_size = game.get_board_size()
    action_size = game.get_action_size()

    print(f"Chess - Board size: {board_size}, Action size: {action_size}")
    print(f"Model will have ~{estimate_model_size()} million parameters")

    # Create the model (this will be VERY large)
    model = ChessModel(
        board_size=board_size, 
        action_size=action_size, 
        device=device,
        num_channels=256,        # Can reduce if memory issues
        num_residual_blocks=19   # Standard AlphaZero depth
    )

    print(f"Model parameters: {count_parameters(model):,}")
    
    # Use existing trainer
    trainer = Trainer(game, model, args)
    
    try:
        trainer.learn()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        # Save current model
        torch.save(model.state_dict(), 'chess_interrupted.pth')
        print("Model saved as chess_interrupted.pth")

def estimate_model_size():
    """Rough estimate of model size"""
    # 256 channels, 19 residual blocks, 4096 action size
    # Roughly 15-20 million parameters
    return 18

def count_parameters(model):
    """Count total model parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def play_chess_game():
    """Simple function to play against the trained model"""
    device = torch.device('cpu')
    game = ChessGame()
    model = ChessModel(game.get_board_size(), game.get_action_size(), device)
    
    try:
        model.load_state_dict(torch.load('chess_latest.pth', map_location=device))
        print("Loaded trained model")
    except:
        print("No trained model found. Train first with main()")
        return
    
    # Implement human vs AI gameplay here
    print("Chess gameplay not implemented yet - requires move input parsing")

if __name__ == '__main__':
    print("🏁 AlphaZero Chess Training")
    print("=" * 50)
    print("⚠️  WARNING: Chess training is extremely resource intensive!")
    print("   - Model has ~18M parameters")  
    print("   - Each game can take 50+ moves")
    print("   - Training will take days/weeks")
    print("   - Consider starting with fewer residual blocks")
    print()
    
    response = input("Continue with training? (y/N): ").lower()
    if response == 'y':
        main()
    else:
        print("Training cancelled. Consider:")
        print("1. Reduce num_residual_blocks in ChessModel")
        print("2. Reduce num_simulations in args")
        print("3. Use smaller num_channels (128 instead of 256)")
        print("4. Start with a smaller chess variant first") 