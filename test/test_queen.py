import chess
import pytest
import torch
from chess import pgn

import chess_ai.games_to_matrices as gtm
import chess_ai.matrices_to_games as mtg


@pytest.mark.parametrize(
    "move_num",
    range(2, 9),
)
def test_queen_up(move_num):
    move = chess.Move.from_uci(f"b1b{move_num}")
    one_hot = torch.zeros(4864)
    one_hot[gtm.move_to_tensor(move)] = 1
    end_move = mtg.tensor_to_move(one_hot, chess.WHITE)
    assert move == end_move


@pytest.mark.parametrize(
    "move_num",
    range(2, 9),
)
def test_queen_down(move_num):
    move = chess.Move.from_uci(f"b8b{9 - move_num}")
    one_hot = torch.zeros(4864)
    one_hot[gtm.move_to_tensor(move)] = 1
    end_move = mtg.tensor_to_move(one_hot, chess.BLACK)
    assert move == end_move


@pytest.mark.parametrize(
    "move_num",
    range(2, 9),
)
def test_queen_left(move_num):
    move = chess.Move.from_uci(f"h1{chr(105 - move_num)}1")
    one_hot = torch.zeros(4864)
    one_hot[gtm.move_to_tensor(move)] = 1
    end_move = mtg.tensor_to_move(one_hot, chess.WHITE)
    assert move == end_move


@pytest.mark.parametrize(
    "move_num",
    range(2, 9),
)
def test_queen_right(move_num):
    move = chess.Move.from_uci(f"a1{chr(96 + move_num)}1")
    one_hot = torch.zeros(4864)
    one_hot[gtm.move_to_tensor(move)] = 1
    end_move = mtg.tensor_to_move(one_hot, chess.WHITE)
    assert move == end_move


@pytest.mark.parametrize(
    "move_num",
    range(2, 9),
)
def test_queen_upleft(move_num):
    move = chess.Move.from_uci(f"h1{chr(105 - move_num)}{move_num}")
    one_hot = torch.zeros(4864)
    one_hot[gtm.move_to_tensor(move)] = 1
    end_move = mtg.tensor_to_move(one_hot, chess.WHITE)
    assert move == end_move


@pytest.mark.parametrize(
    "move_num",
    range(2, 9),
)
def test_queen_upright(move_num):
    move = chess.Move.from_uci(f"a1{chr(96 + move_num)}{move_num}")
    one_hot = torch.zeros(4864)
    one_hot[gtm.move_to_tensor(move)] = 1
    end_move = mtg.tensor_to_move(one_hot, chess.WHITE)
    assert move == end_move


@pytest.mark.parametrize(
    "move_num",
    range(2, 9),
)
def test_queen_downleft(move_num):
    move = chess.Move.from_uci(f"h8{chr(105 - move_num)}{9 - move_num}")
    one_hot = torch.zeros(4864)
    one_hot[gtm.move_to_tensor(move)] = 1
    end_move = mtg.tensor_to_move(one_hot, chess.BLACK)
    assert move == end_move


@pytest.mark.parametrize(
    "move_num",
    range(2, 9),
)
def test_queen_downright(move_num):
    move = chess.Move.from_uci(f"a8{chr(96 + move_num)}{9 - move_num}")
    one_hot = torch.zeros(4864)
    one_hot[gtm.move_to_tensor(move)] = 1
    end_move = mtg.tensor_to_move(one_hot, chess.BLACK)
    assert move == end_move
