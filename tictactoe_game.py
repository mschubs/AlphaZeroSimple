import numpy as np


class TicTacToeGame:
    """
    Standard 3x3 Tic-Tac-Toe game
    """

    def __init__(self):
        self.size = 3

    def get_init_board(self):
        """Return initial empty board"""
        return np.zeros((self.size, self.size), dtype=np.int64)

    def get_board_size(self):
        """Return total board size (flattened)"""
        return self.size * self.size

    def get_action_size(self):
        """Return number of possible actions"""
        return self.size * self.size

    def get_next_state(self, board, player, action):
        """
        Return new board state after player makes action
        Action is 0-8 for positions 0,0 to 2,2 (row-major order)
        """
        b = np.copy(board)
        row, col = action // self.size, action % self.size
        b[row][col] = player
        return (b, -player)

    def has_legal_moves(self, board):
        """Check if any moves are available"""
        return np.any(board == 0)

    def get_valid_moves(self, board):
        """Return binary array of valid moves (1 = valid, 0 = invalid)"""
        valid_moves = []
        for i in range(self.size):
            for j in range(self.size):
                valid_moves.append(1 if board[i][j] == 0 else 0)
        return valid_moves

    def is_win(self, board, player):
        """Check if player has won"""
        # Check rows
        for i in range(self.size):
            if np.all(board[i, :] == player):
                return True
        
        # Check columns
        for j in range(self.size):
            if np.all(board[:, j] == player):
                return True
        
        # Check diagonals
        if np.all(np.diag(board) == player):
            return True
        if np.all(np.diag(np.fliplr(board)) == player):
            return True
        
        return False

    def get_reward_for_player(self, board, player):
        """
        Return None if game not ended
        Return 1 if player wins, -1 if player loses, 0 if draw
        """
        if self.is_win(board, player):
            return 1
        if self.is_win(board, -player):
            return -1
        if not self.has_legal_moves(board):
            return 0  # Draw
        return None  # Game continues

    def get_canonical_board(self, board, player):
        """Return board from player's perspective"""
        return player * board

    def print_board(self, board):
        """Print board in readable format"""
        symbols = {0: '.', 1: 'X', -1: 'O'}
        for i in range(self.size):
            row = ' '.join([symbols[board[i][j]] for j in range(self.size)])
            print(row)
        print() 