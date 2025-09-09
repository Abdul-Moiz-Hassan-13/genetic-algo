import sys
import math
import time
from copy import deepcopy

class ChessPiece:
    def __init__(self, color, piece_type):
        self.color = color
        self.piece_type = piece_type
    
    def __str__(self):
        if self.color == 'white':
            return self.piece_type.upper()
        else:
            return self.piece_type.lower()

class ChessBoard:
    def __init__(self):
        self.board = self.create_board()
        self.en_passant_target = None
        self.castling_rights = {
            'white': {'kingside': True, 'queenside': True},
            'black': {'kingside': True, 'queenside': True}
        }
        self.king_positions = {'white': (7, 4), 'black': (0, 4)}
        self.captured_pieces = {'white': [], 'black': []}
        self.move_history = []

    def create_board(self):
        board = [[None for _ in range(8)] for _ in range(8)]
        
        # Setup pawns
        for i in range(8):
            board[1][i] = ChessPiece('black', 'p')
            board[6][i] = ChessPiece('white', 'p')
        
        # Setup other pieces
        pieces = ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r']
        for i in range(8):
            board[0][i] = ChessPiece('black', pieces[i])
            board[7][i] = ChessPiece('white', pieces[i])
        
        return board

    def display(self):
        print("\n" + "="*40)
        print("    a  b  c  d  e  f  g  h  ")
        print("  ┌────────────────────────┐")
        
        for i in range(8):
            row_num = 8 - i
            row_display = f"{row_num} │"
            
            for j in range(8):
                piece = self.board[i][j]
                piece_str = str(piece) if piece else ' '
                if (i + j) % 2 == 0:
                    row_display += f"\033[48;5;255m {piece_str} \033[0m"
                else:
                    row_display += f"\033[48;5;0m {piece_str} \033[0m"
            
            row_display += "│"
            
            if i == 0:
                row_display += "  Black's Side"
            elif i == 7:
                row_display += "  White's Side"
            
            print(row_display)
        
        print("  └────────────────────────┘")
        print("    a  b  c  d  e  f  g  h  ")

    def move_piece(self, r1, c1, r2, c2):
        piece = self.board[r1][c1]
        if not piece:
            return False

        # Track captures
        target = self.board[r2][c2]
        if target and target.color != piece.color:
            self.captured_pieces[piece.color].append(target)

        # Handle castling
        if piece.piece_type == 'k' and abs(c1 - c2) == 2:
            return self.execute_castling(r1, c1, r2, c2)

        # Handle en passant capture
        if (piece.piece_type == 'p' and 
            (r2, c2) == self.en_passant_target and 
            abs(c1 - c2) == 1):
            captured_pawn = self.board[r1][c2]
            self.captured_pieces[piece.color].append(captured_pawn)
            self.board[r1][c2] = None

        # Handle promotion
        if piece.piece_type == 'p' and (r2 == 0 or r2 == 7):
            piece.piece_type = self.get_promotion_choice(piece.color)

        # Update king position
        if piece.piece_type == 'k':
            self.king_positions[piece.color] = (r2, c2)
            self.castling_rights[piece.color] = {'kingside': False, 'queenside': False}

        # Update rook castling rights
        if piece.piece_type == 'r':
            if piece.color == 'white':
                if r1 == 7 and c1 == 0:
                    self.castling_rights['white']['queenside'] = False
                elif r1 == 7 and c1 == 7:
                    self.castling_rights['white']['kingside'] = False
            else:
                if r1 == 0 and c1 == 0:
                    self.castling_rights['black']['queenside'] = False
                elif r1 == 0 and c1 == 7:
                    self.castling_rights['black']['kingside'] = False

        # Update en passant target
        if piece.piece_type == 'p' and abs(r2 - r1) == 2:
            self.en_passant_target = (r1 + (r2 - r1)//2, c1)
        else:
            self.en_passant_target = None

        # Execute the move
        self.board[r2][c2] = piece
        self.board[r1][c1] = None
        
        # Record move
        move_notation = self.get_move_notation(r1, c1, r2, c2, piece, target)
        self.move_history.append(move_notation)
        
        return True

    def get_move_notation(self, r1, c1, r2, c2, piece, target):
     start = coords_to_algebraic(r1, c1)
     end = coords_to_algebraic(r2, c2)
    
     # Castling
     if piece.piece_type == 'k' and abs(c1 - c2) == 2:
        return "O-O" if c2 > c1 else "O-O-O"
    
    # Capture
     if target:
         if piece.piece_type == 'p':
            return f"{start}x{end}"
         return f"{str(piece).upper()}{start}x{end}"
    
    # Pawn move
     if piece.piece_type == 'p':
         return f"{start}-{end}"
    
    # Normal move
     return f"{str(piece).upper()}{start}-{end}"

    def execute_castling(self, r1, c1, r2, c2):
        king = self.board[r1][c1]
        direction = 1 if c2 > c1 else -1
        
        # Move king
        self.board[r2][c2] = king
        self.board[r1][c1] = None
        self.king_positions[king.color] = (r2, c2)
        
        # Move rook
        rook_col = 7 if direction == 1 else 0
        new_rook_col = 5 if direction == 1 else 3
        rook = self.board[r1][rook_col]
        self.board[r1][new_rook_col] = rook
        self.board[r1][rook_col] = None
        
        self.castling_rights[king.color] = {'kingside': False, 'queenside': False}
        
        # Record move
        move_notation = "O-O" if direction == 1 else "O-O-O"
        self.move_history.append(f"{king.color.capitalize()}: {move_notation}")
        
        return True

    def get_promotion_choice(self, color):
        while True:
            choice = input(f"Promote pawn to (Q/R/B/N): ").strip().lower()
            if choice in ['q', 'r', 'b', 'n']:
                return choice
            print("Invalid choice. Please enter Q, R, B, or N.")

    def copy(self):
        new_board = ChessBoard()
        new_board.board = deepcopy(self.board)
        new_board.en_passant_target = self.en_passant_target
        new_board.castling_rights = deepcopy(self.castling_rights)
        new_board.king_positions = deepcopy(self.king_positions)
        new_board.captured_pieces = deepcopy(self.captured_pieces)
        new_board.move_history = deepcopy(self.move_history)
        return new_board

def algebraic_to_coords(alg):
    col = ord(alg[0]) - ord('a')
    row = 8 - int(alg[1])
    return row, col

def coords_to_algebraic(row, col):
    return chr(ord('a') + col) + str(8 - row)

def is_square_under_attack(board, row, col, color):
    opponent = 'black' if color == 'white' else 'white'
    
    # Check for knight attacks
    knight_moves = [(-2,-1), (-2,1), (-1,-2), (-1,2),
                    (1,-2), (1,2), (2,-1), (2,1)]
    for dr, dc in knight_moves:
        r, c = row + dr, col + dc
        if 0 <= r < 8 and 0 <= c < 8:
            piece = board.board[r][c]
            if piece and piece.color == opponent and piece.piece_type == 'n':
                return True
    
    # Check for sliding pieces
    directions = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]
    
    for dr, dc in directions:
        r, c = row + dr, col + dc
        while 0 <= r < 8 and 0 <= c < 8:
            piece = board.board[r][c]
            if piece:
                if piece.color == opponent:
                    if (abs(dr) + abs(dc) == 1 and piece.piece_type in ['r', 'q']):
                        return True
                    if (abs(dr) == 1 and abs(dc) == 1 and piece.piece_type in ['b', 'q']):
                        return True
                    if piece.piece_type == 'k' and abs(r-row) <= 1 and abs(c-col) <= 1:
                        return True
                break
            r += dr
            c += dc
    
    # Check for pawn attacks
    pawn_dir = -1 if opponent == 'white' else 1
    for dc in [-1, 1]:
        r = row + pawn_dir
        c = col + dc
        if 0 <= r < 8 and 0 <= c < 8:
            piece = board.board[r][c]
            if piece and piece.color == opponent and piece.piece_type == 'p':
                return True
    
    return False

def is_in_check(board, color):
    king_pos = board.king_positions[color]
    return is_square_under_attack(board, king_pos[0], king_pos[1], color)

def generate_legal_moves(board, color):
    legal_moves = []
    
    for r1 in range(8):
        for c1 in range(8):
            piece = board.board[r1][c1]
            if not piece or piece.color != color:
                continue
            
            if piece.piece_type == 'p':
                moves = generate_pawn_moves(board, r1, c1, color)
            elif piece.piece_type == 'n':
                moves = generate_knight_moves(board, r1, c1, color)
            elif piece.piece_type == 'b':
                moves = generate_bishop_moves(board, r1, c1, color)
            elif piece.piece_type == 'r':
                moves = generate_rook_moves(board, r1, c1, color)
            elif piece.piece_type == 'q':
                moves = generate_queen_moves(board, r1, c1, color)
            elif piece.piece_type == 'k':
                moves = generate_king_moves(board, r1, c1, color)
            
            for r2, c2 in moves:
                temp_board = board.copy()
                temp_board.move_piece(r1, c1, r2, c2)
                if not is_in_check(temp_board, color):
                    legal_moves.append((r1, c1, r2, c2))
    
    return legal_moves

def generate_pawn_moves(board, r1, c1, color):
    moves = []
    direction = -1 if color == 'white' else 1
    
    # Forward move
    if 0 <= r1 + direction < 8 and not board.board[r1 + direction][c1]:
        moves.append((r1 + direction, c1))
        if ((color == 'white' and r1 == 6) or (color == 'black' and r1 == 1)) and \
           not board.board[r1 + 2*direction][c1]:
            moves.append((r1 + 2*direction, c1))
    
    # Captures
    for dc in [-1, 1]:
        c2 = c1 + dc
        if 0 <= c2 < 8:
            if 0 <= r1 + direction < 8 and board.board[r1 + direction][c2] and \
               board.board[r1 + direction][c2].color != color:
                moves.append((r1 + direction, c2))
            if board.en_passant_target and (r1 + direction, c2) == board.en_passant_target:
                moves.append((r1 + direction, c2))
    
    return moves

def generate_knight_moves(board, r1, c1, color):
    moves = []
    knight_moves = [(-2,-1), (-2,1), (-1,-2), (-1,2),
                    (1,-2), (1,2), (2,-1), (2,1)]
    
    for dr, dc in knight_moves:
        r2, c2 = r1 + dr, c1 + dc
        if 0 <= r2 < 8 and 0 <= c2 < 8:
            piece = board.board[r2][c2]
            if not piece or piece.color != color:
                moves.append((r2, c2))
    
    return moves

def generate_bishop_moves(board, r1, c1, color):
    moves = []
    directions = [(-1,-1), (-1,1), (1,-1), (1,1)]
    
    for dr, dc in directions:
        r2, c2 = r1 + dr, c1 + dc
        while 0 <= r2 < 8 and 0 <= c2 < 8:
            piece = board.board[r2][c2]
            if piece:
                if piece.color != color:
                    moves.append((r2, c2))
                break
            moves.append((r2, c2))
            r2 += dr
            c2 += dc
    
    return moves

def generate_rook_moves(board, r1, c1, color):
    moves = []
    directions = [(-1,0), (1,0), (0,-1), (0,1)]
    
    for dr, dc in directions:
        r2, c2 = r1 + dr, c1 + dc
        while 0 <= r2 < 8 and 0 <= c2 < 8:
            piece = board.board[r2][c2]
            if piece:
                if piece.color != color:
                    moves.append((r2, c2))
                break
            moves.append((r2, c2))
            r2 += dr
            c2 += dc
    
    return moves

def generate_queen_moves(board, r1, c1, color):
    return generate_rook_moves(board, r1, c1, color) + \
           generate_bishop_moves(board, r1, c1, color)

def generate_king_moves(board, r1, c1, color):
    moves = []
    
    # Normal king moves
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            r2, c2 = r1 + dr, c1 + dc
            if 0 <= r2 < 8 and 0 <= c2 < 8:
                piece = board.board[r2][c2]
                if not piece or piece.color != color:
                    moves.append((r2, c2))
    
    # Castling
    if not is_in_check(board, color):
        for side in ['kingside', 'queenside']:
            if board.castling_rights[color][side]:
                can_castle = True
                if side == 'kingside':
                    rook_col = 7
                    step = 1
                else:
                    rook_col = 0
                    step = -1
                
                for c in range(c1 + step, rook_col, step):
                    if board.board[r1][c]:
                        can_castle = False
                        break
                
                if can_castle:
                    for c in range(c1, c1 + 2*step, step):
                        if is_square_under_attack(board, r1, c, color):
                            can_castle = False
                            break
                
                if can_castle:
                    moves.append((r1, c1 + 2*step))
    
    return moves

def evaluate_board(board):
    piece_values = {'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'k': 0}
    score = 0
    
    # Material score
    for row in range(8):
        for col in range(8):
            piece = board.board[row][col]
            if piece:
                value = piece_values[piece.piece_type]
                if piece.color == 'white':
                    score += value
                else:
                    score -= value
    
    # Positional bonuses
    center_squares = [(3,3), (3,4), (4,3), (4,4)]
    for row, col in center_squares:
        piece = board.board[row][col]
        if piece and piece.color == 'white':
            score += 0.1
        elif piece and piece.color == 'black':
            score -= 0.1
    
    # Mobility bonus
    white_moves = len(generate_legal_moves(board, 'white'))
    black_moves = len(generate_legal_moves(board, 'black'))
    score += 0.01 * (white_moves - black_moves)
    
    return score

def alpha_beta_search(board, depth, alpha, beta, maximizing_player):
    if depth == 0 or is_checkmate(board, 'white' if maximizing_player else 'black'):
        return evaluate_board(board), None
    
    best_move = None
    legal_moves = generate_legal_moves(board, 'white' if maximizing_player else 'black')
    
    if maximizing_player:
        max_eval = -math.inf
        for r1, c1, r2, c2 in legal_moves:
            new_board = board.copy()
            new_board.move_piece(r1, c1, r2, c2)
            eval, _ = alpha_beta_search(new_board, depth-1, alpha, beta, False)
            if eval > max_eval:
                max_eval = eval
                best_move = (r1, c1, r2, c2)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = math.inf
        for r1, c1, r2, c2 in legal_moves:
            new_board = board.copy()
            new_board.move_piece(r1, c1, r2, c2)
            eval, _ = alpha_beta_search(new_board, depth-1, alpha, beta, True)
            if eval < min_eval:
                min_eval = eval
                best_move = (r1, c1, r2, c2)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval, best_move

def is_checkmate(board, color):
    if not is_in_check(board, color):
        return False
    return len(generate_legal_moves(board, color)) == 0

def is_stalemate(board, color):
    if is_in_check(board, color):
        return False
    return len(generate_legal_moves(board, color)) == 0

def is_insufficient_material(board):
    pieces = {'white': [], 'black': []}
    for row in range(8):
        for col in range(8):
            piece = board.board[row][col]
            if piece:
                pieces[piece.color].append(piece.piece_type)
    
    # King vs King
    if len(pieces['white']) == 1 and len(pieces['black']) == 1:
        return True
    
    # King + bishop/knight vs King
    for color in ['white', 'black']:
        if len(pieces[color]) == 2 and any(p in pieces[color] for p in ['b', 'n']):
            return True
    
    # Both sides have only king + bishop with bishops on same color
    if (len(pieces['white']) == 2 and 'b' in pieces['white'] and
        len(pieces['black']) == 2 and 'b' in pieces['black']):
        return True
    
    return False

def display_move_history(move_history):
    print("\nGame Move History:")
    for i in range(0, len(move_history), 2):
        white_move = move_history[i] if i < len(move_history) else ""
        black_move = move_history[i+1] if i+1 < len(move_history) else ""
        
        # Convert notation to show from-to squares
        if white_move:
            if white_move.startswith("O-O"):
                white_display = white_move
            else:
                white_display = f"{white_move[:2]}-{white_move[-2:]}" if '-' in white_move else white_move
                white_display = white_display.replace('x', '-')  # Replace captures with dash
        
        if black_move:
            if black_move.startswith("O-O"):
                black_display = black_move
            else:
                black_display = f"{black_move[:2]}-{black_move[-2:]}" if '-' in black_move else black_move
                black_display = black_display.replace('x', '-')  # Replace captures with dash
        
        print(f"{i//2 + 1}. {white_display.ljust(10)} {black_display}")

def main():
    print("♟️ Welcome! White pieces are WHITE, Black pieces are BLACK.\n")
    board = ChessBoard()
    side_to_move = 'white'

    while True:
        board.display()
        print(f"\nCaptured by White: {[str(p) for p in board.captured_pieces['white']]}")
        print(f"Captured by Black: {[str(p) for p in board.captured_pieces['black']]}")

        # Check end conditions
        if is_checkmate(board, side_to_move):
            winner = 'Black' if side_to_move == 'white' else 'White'
            print(f"\nCheckmate! {winner} wins!\n")
            display_move_history(board.move_history)
            break
        if is_stalemate(board, side_to_move) or is_insufficient_material(board):
            print("\nDraw! Game ended.\n")
            display_move_history(board.move_history)
            break

        if side_to_move == 'white':
            # Human plays White
            print("\nWhite's turn. Enter your move (e.g. e2e4). 'quit' to exit.")
            user_in = input("Your move: ").strip().lower()
            if user_in in ["quit", "exit"]:
                print("\nGame ended by user.")
                display_move_history(board.move_history)
                sys.exit(0)
            
            if len(user_in) != 4:
                print("Invalid format! Try e2e4.")
                continue
            
            try:
                r1, c1 = algebraic_to_coords(user_in[:2])
                r2, c2 = algebraic_to_coords(user_in[2:])
            except:
                print("Invalid coordinates! Try e2e4.")
                continue
            
            legal_moves = generate_legal_moves(board, 'white')
            if (r1, c1, r2, c2) not in legal_moves:
                print("Illegal move! Try again.")
                continue
            
            board.move_piece(r1, c1, r2, c2)
            side_to_move = 'black'
        else:
            # AI plays Black
            print("\nBlack AI thinking...")
            depth = 2
            start_time = time.time()
            _, best_move = alpha_beta_search(board, depth, -math.inf, math.inf, False)
            end_time = time.time()
            
            if best_move is None:
                print("AI found no legal moves!")
                continue
            
            r1, c1, r2, c2 = best_move
            board.move_piece(r1, c1, r2, c2)
            move_notation = board.get_move_notation(r1, c1, r2, c2, 
                                                 board.board[r2][c2], 
                                                 None if (r1,c1,r2,c2) not in board.move_history else board.board[r2][c2])
            print(f"Black plays: {move_notation} (thought for {end_time-start_time:.1f}s)")
            side_to_move = 'white'

if __name__ == "__main__":
    main()