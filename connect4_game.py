import numpy as np


class Connect4Game:
    """
    Connect 4 game: 6x7 board, players drop pieces into columns
    Goal: Get 4 pieces in a row (horizontally, vertically, or diagonally)
    """

    def __init__(self):
        self.height = 6
        self.width = 7

    def get_init_board(self):
        """Return initial empty board"""
        return np.zeros((self.height, self.width), dtype=np.int64)

    def get_board_size(self):
        """Return total board size (flattened)"""
        return self.height * self.width

    def get_action_size(self):
        """Return number of possible actions (one per column)"""
        return self.width

    def get_next_state(self, board, player, action):
        """
        Return new board state after player drops piece in column
        Action is 0-6 for columns 0-6
        """
        b = np.copy(board)
        
        # Find the lowest empty row in the chosen column
        for row in range(self.height - 1, -1, -1):
            if b[row][action] == 0:
                b[row][action] = player
                break
        
        return (b, -player)

    def has_legal_moves(self, board):
        """Check if any moves are available"""
        # Check if any column has space in the top row
        return np.any(board[0, :] == 0)

    def get_valid_moves(self, board):
        """Return binary array of valid moves (1 = valid, 0 = invalid)"""
        valid_moves = []
        for col in range(self.width):
            # Column is valid if top row is empty
            valid_moves.append(1 if board[0][col] == 0 else 0)
        return valid_moves

    def is_win(self, board, player):
        """Check if player has won (4 in a row)"""
        
        # Check horizontal
        for row in range(self.height):
            for col in range(self.width - 3):
                if all(board[row][col + i] == player for i in range(4)):
                    return True
        
        # Check vertical
        for row in range(self.height - 3):
            for col in range(self.width):
                if all(board[row + i][col] == player for i in range(4)):
                    return True
        
        # Check diagonal (top-left to bottom-right)
        for row in range(self.height - 3):
            for col in range(self.width - 3):
                if all(board[row + i][col + i] == player for i in range(4)):
                    return True
        
        # Check diagonal (top-right to bottom-left)
        for row in range(self.height - 3):
            for col in range(3, self.width):
                if all(board[row + i][col - i] == player for i in range(4)):
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
        
        print("  0 1 2 3 4 5 6")  # Column numbers
        for i in range(self.height):
            row = ' '.join([symbols[board[i][j]] for j in range(self.width)])
            print(f"{i} {row}")
        print()

    def get_symmetries(self, board, pi):
        """
        Return list of symmetries for Connect 4
        Only horizontal flip is valid (vertical flip would break gravity)
        """
        symmetries = []
        
        # Original
        symmetries.append((board, pi))
        
        # Horizontal flip
        flipped_board = np.fliplr(board)
        flipped_pi = pi[::-1]  # Reverse the policy vector
        symmetries.append((flipped_board, flipped_pi))
        
        return symmetries 