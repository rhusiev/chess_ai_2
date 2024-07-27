import chess
import torch
from chess import pgn

import chess_ai.games_to_matrices as gtm
import chess_ai.matrices_to_games as mtg


def test_promotion_up():
    move = chess.Move.from_uci("a7a8q")
    one_hot = torch.zeros(4864)
    one_hot[gtm.move_to_tensor(move)] = 1
    end_move = mtg.tensor_to_move(one_hot, chess.WHITE)
    assert move == end_move


def test_promotion_upright():
    move = chess.Move.from_uci("b7c8n")
    one_hot = torch.zeros(4864)
    one_hot[gtm.move_to_tensor(move)] = 1
    end_move = mtg.tensor_to_move(one_hot, chess.WHITE)
    assert move == end_move


def test_promotion_upleft():
    move = chess.Move.from_uci("h7g8b")
    one_hot = torch.zeros(4864)
    one_hot[gtm.move_to_tensor(move)] = 1
    end_move = mtg.tensor_to_move(one_hot, chess.WHITE)
    assert move == end_move


def test_promotion_down():
    move = chess.Move.from_uci("d2d1r")
    one_hot = torch.zeros(4864)
    one_hot[gtm.move_to_tensor(move)] = 1
    end_move = mtg.tensor_to_move(one_hot, chess.BLACK)
    assert move == end_move


def test_promotion_downright():
    move = chess.Move.from_uci("e2f1n")
    one_hot = torch.zeros(4864)
    one_hot[gtm.move_to_tensor(move)] = 1
    end_move = mtg.tensor_to_move(one_hot, chess.BLACK)
    assert move == end_move


def test_promotion_downleft():
    move = chess.Move.from_uci("g2f1q")
    one_hot = torch.zeros(4864)
    one_hot[gtm.move_to_tensor(move)] = 1
    end_move = mtg.tensor_to_move(one_hot, chess.BLACK)
    assert move == end_move
