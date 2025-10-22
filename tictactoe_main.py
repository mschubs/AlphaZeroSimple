import torch

from tictactoe_game import TicTacToeGame
from tictactoe_model import TicTacToeModel
from trainer import Trainer

# Use CPU for now to avoid MPS numerical issues
device = torch.device('cpu')
print("Using CPU (avoiding MPS numerical instability)")

args = {
    'batch_size': 64,
    'numIters': 1,                                # Total number of training iterations
    'num_simulations': 100,                         # Total number of MCTS simulations to run when deciding on a move to play
    'numEps': 50,                                   # Number of full games (episodes) to run during each iteration
    'numItersForTrainExamplesHistory': 20,
    'epochs': 4,                                    # Number of epochs of training per iteration
    'checkpoint_path': 'tictactoe_latest_test1.pth'       # location to save latest set of weights
}

game = TicTacToeGame()
board_size = game.get_board_size()
action_size = game.get_action_size()

model = TicTacToeModel(board_size, action_size, device)

trainer = Trainer(game, model, args)
trainer.learn() 