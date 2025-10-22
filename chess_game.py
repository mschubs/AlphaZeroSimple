import numpy as np
import chess
import chess.engine

class ChessGame:
    """
    Chess game implementation for AlphaZero
    Uses python-chess library for move generation and validation
    """
    
    def __init__(self):
        self.board_size = 8 * 8 * 12  # 8x8 board, 12 piece types per square
        # Action encoding: from_square * 64 + to_square (simplified)
        # More sophisticated: from_square * 73 + move_type (includes promotions, etc.)
        self.action_size = 64 * 64  # Simplified: 4096 possible moves
        
    def get_init_board(self):
        """Return initial chess position as numpy array"""
        board = chess.Board()
        return self._board_to_array(board)
    
    def get_board_size(self):
        return self.board_size
    
    def get_action_size(self):
        return self.action_size
    
    def _board_to_array(self, board):
        """
        Convert chess.Board to numpy array representation
        Shape: (8, 8, 12) - 12 channels for piece types
        Channels 0-5: White pieces (P,N,B,R,Q,K)
        Channels 6-11: Black pieces (p,n,b,r,q,k)
        """
        array = np.zeros((8, 8, 12), dtype=np.float32)
        
        piece_map = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                row = 7 - (square // 8)  # Flip vertically
                col = square % 8
                piece_type = piece_map[piece.piece_type]
                if piece.color == chess.WHITE:
                    array[row, col, piece_type] = 1
                else:
                    array[row, col, piece_type + 6] = 1
        
        return array.flatten()  # Flatten for neural network input
    
    def _array_to_board(self, array):
        """Convert numpy array back to chess.Board"""
        # This is more complex - you'd reconstruct the board state
        # For now, we'll track the actual board state separately
        pass
    
    def _encode_move(self, move):
        """Encode chess.Move to action index"""
        return move.from_square * 64 + move.to_square
    
    def _decode_action(self, action):
        """Decode action index to from_square, to_square"""
        from_square = action // 64
        to_square = action % 64
        return from_square, to_square
    
    def get_next_state(self, board_array, player, action):
        """Apply move and return new state"""
        # You'll need to maintain a chess.Board object alongside the array
        # This is a simplified version - real implementation needs state tracking
        
        board = self._reconstruct_board(board_array, player)
        from_square, to_square = self._decode_action(action)
        
        # Find the actual move from legal moves
        legal_moves = list(board.legal_moves)
        move = None
        for m in legal_moves:
            if m.from_square == from_square and m.to_square == to_square:
                move = m
                break
        
        if move:
            board.push(move)
            return (self._board_to_array(board), -player)
        else:
            # Invalid move - this shouldn't happen with proper masking
            return (board_array, player)
    
    def _reconstruct_board(self, board_array, player):
        """Reconstruct chess.Board from array representation"""
        # This is complex - you might want to maintain board state separately
        # For now, return a new board (placeholder)
        return chess.Board()
    
    def has_legal_moves(self, board_array):
        """Check if current position has legal moves"""
        board = self._reconstruct_board(board_array, 1)  # Player doesn't matter for legal moves check
        return not board.is_game_over()
    
    def get_valid_moves(self, board_array):
        """Return binary array of valid moves"""
        board = self._reconstruct_board(board_array, 1)
        valid_moves = [0] * self.action_size
        
        for move in board.legal_moves:
            action = self._encode_move(move)
            if 0 <= action < self.action_size:
                valid_moves[action] = 1
        
        return valid_moves
    
    def is_win(self, board_array, player):
        """Check if player has won (checkmate opponent)"""
        board = self._reconstruct_board(board_array, player)
        
        if board.is_checkmate():
            # The player to move is checkmated, so the other player won
            return board.turn != (player == 1)  # Convert player encoding
        return False
    
    def get_reward_for_player(self, board_array, player):
        """Return game outcome from player's perspective"""
        board = self._reconstruct_board(board_array, player)
        
        if board.is_checkmate():
            # Player to move is checkmated
            if board.turn == (player == 1):
                return -1  # Player lost
            else:
                return 1   # Player won
        elif board.is_stalemate() or board.is_insufficient_material() or board.is_fifty_moves():
            return 0  # Draw
        elif self.has_legal_moves(board_array):
            return None  # Game continues
        else:
            return 0  # Some other draw condition
    
    def get_canonical_board(self, board_array, player):
        """Return board from current player's perspective"""
        if player == 1:
            return board_array
        else:
            # For chess, you might want to flip the board for black player
            # This is optional - many implementations don't do this
            return self._flip_board(board_array)
    
    def _flip_board(self, board_array):
        """Flip board for black player's perspective"""
        # Reshape, flip, swap piece colors, flatten
        board_3d = board_array.reshape(8, 8, 12)
        flipped = np.flip(board_3d, axis=0)  # Flip vertically
        
        # Swap white and black pieces
        temp = flipped[:, :, :6].copy()
        flipped[:, :, :6] = flipped[:, :, 6:]
        flipped[:, :, 6:] = temp
        
        return flipped.flatten()
    
    def print_board(self, board_array):
        """Print human-readable board"""
        board = self._reconstruct_board(board_array, 1)
        print(board) 