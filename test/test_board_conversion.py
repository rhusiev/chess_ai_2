import chess
import torch
from chess import pgn

import chess_ai.games_to_matrices as gtm
import chess_ai.matrices_to_games as mtg


def test_basic():
    game = pgn.Game()
    starting_board = game.board()
    tensor = gtm.game_to_tensor(game)[0]
    end_board = mtg.tensor_to_board(tensor)
    assert str(starting_board) == str(end_board)


def test_ingame1():
    game = pgn.Game()
    game.add_main_variation(chess.Move.from_uci("e2e4"))
    game.add_main_variation(chess.Move.from_uci("e7e5"))
    game.add_main_variation(chess.Move.from_uci("g1f3"))
    game.add_main_variation(chess.Move.from_uci("b8c6"))
    game.add_main_variation(chess.Move.from_uci("f1c4"))
    game.add_main_variation(chess.Move.from_uci("g8f6"))
    starting_board = game.board()
    tensor = gtm.game_to_tensor(game)[0]
    end_board = mtg.tensor_to_board(tensor)
    assert str(starting_board) == str(end_board)


def test_ingame2():
    game = pgn.Game.from_board(
        chess.Board(fen="rnbq1bnr/p2Pkppp/4p3/8/8/8/1PPPPPPP/RNBQKBNR w KQ - 1 5")
    )
    starting_board = game.board()
    tensor = gtm.game_to_tensor(game)[0]
    end_board = mtg.tensor_to_board(tensor)
    assert str(starting_board) == str(end_board)
