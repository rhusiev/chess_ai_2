import chess
import torch


def tensor_to_board(tensor):
    board = chess.Board()
    board.clear()
    idx_to_piece = [
        chess.PAWN,
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK,
        chess.QUEEN,
        chess.KING,
    ]
    for row in range(8):
        for col in range(8):
            for piece_idx in range(6):
                white_offset = 0
                piece_is_here = tensor[row][col][piece_idx + white_offset]
                if piece_is_here:
                    piece = idx_to_piece[piece_idx]
                    board.set_piece_at(
                        chess.square(col, row), chess.Piece(piece, chess.WHITE)
                    )

                black_offset = 6
                piece_is_here = tensor[row][col][piece_idx + black_offset]
                if piece_is_here:
                    piece = idx_to_piece[piece_idx]
                    board.set_piece_at(
                        chess.square(col, row), chess.Piece(piece, chess.BLACK)
                    )
    return board


def info_to_move(from_square, to_square, promotion=None):
    move = chess.Move(from_square, to_square)
    if promotion:
        if promotion == chess.QUEEN:
            move.promotion = chess.QUEEN
        elif promotion == chess.BISHOP:
            move.promotion = chess.BISHOP
        elif promotion == chess.ROOK:
            move.promotion = chess.ROOK
        elif promotion == chess.KNIGHT:
            move.promotion = chess.KNIGHT
    return move


def tensor_to_move(tensor, color):
    tensor_8_8_76_form = tensor.view(8, 8, 76)
    for row in range(8):
        for col in range(8):
            promotion = None
            from_square = chess.square(col, row)
            for i in range(64, 67):
                if tensor_8_8_76_form[row][col][i]:
                    promotion = chess.KNIGHT
                    to_square = chess.square(
                        col - 65 + i, row + 1 if color == chess.WHITE else row - 1
                    )
                    return info_to_move(from_square, to_square, promotion)
            for i in range(67, 70):
                if tensor_8_8_76_form[row][col][i]:
                    promotion = chess.ROOK
                    to_square = chess.square(
                        col - 68 + i, row + 1 if color == chess.WHITE else row - 1
                    )
                    return info_to_move(from_square, to_square, promotion)
            for i in range(70, 73):
                if tensor_8_8_76_form[row][col][i]:
                    promotion = chess.BISHOP
                    to_square = chess.square(
                        col - 71 + i, row + 1 if color == chess.WHITE else row - 1
                    )
                    return info_to_move(from_square, to_square, promotion)
            for i in range(73, 76):
                if tensor_8_8_76_form[row][col][i]:
                    promotion = chess.QUEEN
                    to_square = chess.square(
                        col - 74 + i, row + 1 if color == chess.WHITE else row - 1
                    )
                    return info_to_move(from_square, to_square, promotion)
            for i in range(7, 56, 8):
                if tensor_8_8_76_form[row][col][i]:
                    to_square = chess.square(col + i // 8 + 1, row + i // 8 + 1)
                    return info_to_move(from_square, to_square)
            for i in range(3, 52, 8):
                if tensor_8_8_76_form[row][col][i]:
                    to_square = chess.square(col - i // 8 - 1, row - i // 8 - 1)
                    return info_to_move(from_square, to_square)
            for i in range(1, 50, 8):
                if tensor_8_8_76_form[row][col][i]:
                    to_square = chess.square(col - i // 8 - 1, row + i // 8 + 1)
                    return info_to_move(from_square, to_square)
            for i in range(5, 54, 8):
                if tensor_8_8_76_form[row][col][i]:
                    to_square = chess.square(col + i // 8 + 1, row - i // 8 - 1)
                    return info_to_move(from_square, to_square)
            for i in range(6, 55, 8):
                if tensor_8_8_76_form[row][col][i]:
                    to_square = chess.square(col + i // 8 + 1, row)
                    return info_to_move(from_square, to_square)
            for i in range(2, 51, 8):
                if tensor_8_8_76_form[row][col][i]:
                    to_square = chess.square(col - i // 8 - 1, row)
                    return info_to_move(from_square, to_square)
            for i in range(0, 49, 8):
                if tensor_8_8_76_form[row][col][i]:
                    to_square = chess.square(col, row + i // 8 + 1)
                    return info_to_move(from_square, to_square)
            for i in range(4, 53, 8):
                if tensor_8_8_76_form[row][col][i]:
                    to_square = chess.square(col, row - i // 8 - 1)
                    return info_to_move(from_square, to_square)
            if tensor_8_8_76_form[row][col][56]:
                to_square = chess.square(col - 1, row + 2)
                return info_to_move(from_square, to_square)
            if tensor_8_8_76_form[row][col][57]:
                to_square = chess.square(col - 2, row + 1)
                return info_to_move(from_square, to_square)
            if tensor_8_8_76_form[row][col][58]:
                to_square = chess.square(col - 2, row - 1)
                return info_to_move(from_square, to_square)
            if tensor_8_8_76_form[row][col][59]:
                to_square = chess.square(col - 1, row - 2)
                return info_to_move(from_square, to_square)
            if tensor_8_8_76_form[row][col][60]:
                to_square = chess.square(col + 1, row - 2)
                return info_to_move(from_square, to_square)
            if tensor_8_8_76_form[row][col][61]:
                to_square = chess.square(col + 2, row - 1)
                return info_to_move(from_square, to_square)
            if tensor_8_8_76_form[row][col][62]:
                to_square = chess.square(col + 2, row + 1)
                return info_to_move(from_square, to_square)
            if tensor_8_8_76_form[row][col][63]:
                to_square = chess.square(col + 1, row + 2)
                return info_to_move(from_square, to_square)
