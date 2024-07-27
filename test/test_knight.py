import chess
import torch
from chess import pgn

import chess_ai.games_to_matrices as gtm
import chess_ai.matrices_to_games as mtg


def test_knight_upleft():
    move = chess.Move.from_uci("e4d6")
    one_hot = torch.zeros(4864)
    one_hot[gtm.move_to_tensor(move)] = 1
    end_move = mtg.tensor_to_move(one_hot, chess.WHITE)
    assert move == end_move


def test_knight_leftup():
    move = chess.Move.from_uci("f4d5")
    one_hot = torch.zeros(4864)
    one_hot[gtm.move_to_tensor(move)] = 1
    end_move = mtg.tensor_to_move(one_hot, chess.WHITE)
    assert move == end_move


def test_knight_leftdown():
    move = chess.Move.from_uci("e5c4")
    one_hot = torch.zeros(4864)
    one_hot[gtm.move_to_tensor(move)] = 1
    end_move = mtg.tensor_to_move(one_hot, chess.WHITE)
    assert move == end_move


def test_knight_downleft():
    move = chess.Move.from_uci("c5b3")
    one_hot = torch.zeros(4864)
    one_hot[gtm.move_to_tensor(move)] = 1
    end_move = mtg.tensor_to_move(one_hot, chess.WHITE)
    assert move == end_move


def test_knight_downright():
    move = chess.Move.from_uci("g4h2")
    one_hot = torch.zeros(4864)
    one_hot[gtm.move_to_tensor(move)] = 1
    end_move = mtg.tensor_to_move(one_hot, chess.WHITE)
    assert move == end_move


def test_knight_rightdown():
    move = chess.Move.from_uci("a2c1")
    one_hot = torch.zeros(4864)
    one_hot[gtm.move_to_tensor(move)] = 1
    end_move = mtg.tensor_to_move(one_hot, chess.WHITE)
    assert move == end_move


def test_knight_rightup():
    move = chess.Move.from_uci("b1d2")
    one_hot = torch.zeros(4864)
    one_hot[gtm.move_to_tensor(move)] = 1
    end_move = mtg.tensor_to_move(one_hot, chess.WHITE)
    assert move == end_move


def test_knight_upright():
    move = chess.Move.from_uci("f6g8")
    one_hot = torch.zeros(4864)
    one_hot[gtm.move_to_tensor(move)] = 1
    end_move = mtg.tensor_to_move(one_hot, chess.WHITE)
    assert move == end_move
