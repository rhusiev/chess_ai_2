import time
import chess
import torch
from chess import pgn

from chess_ai.games_to_matrices import game_to_tensor
from chess_ai.load_saved import load_model
from chess_ai.matrices_to_games import tensor_to_move

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = load_model("models/model_parameters_800_1200_epoch2_batch8000.pth")
model.to(device)

fen = input("Enter starting FEN: ")
board = chess.Board(fen=fen)
game = pgn.Game.from_board(board)

print(game.board())


def get_move():
    move_str = input("Enter move: ")
    if move_str == "q":
        exit()
    try:
        move = chess.Move.from_uci(move_str)
    except ValueError:
        print("Invalid move")
        return get_move()
    if move not in game.board().legal_moves:
        print("Illegal move")
        return get_move()
    return move


def get_bot_move():
    state, consts = game_to_tensor(game)
    state = (
        state.permute(2, 0, 1).unsqueeze(0).to(device)
    )  # Adds a batch dimension and moves to the device
    consts = consts.unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        raw_output = model(state, consts)
        probabilities = torch.softmax(raw_output, dim=1)
        # sort by probability and get the first legal move
        _, sorted_indices = torch.sort(probabilities, descending=True)
        for i in range(sorted_indices.size(1)):
            one_hot = torch.zeros(sorted_indices.size(1))
            one_hot[sorted_indices[0][i]] = 1
            move = tensor_to_move(one_hot, game.board().turn)
            if move in game.board().legal_moves:
                return move
            print("Bot move was illegal; trying next move")
    return None


def play_with_bot():
    global game
    while not game.board().is_game_over():
        move = get_bot_move()
        if not move:
            print("No legal moves found")
            break
        game.add_main_variation(move)
        game = game.next()
        if game.board().is_game_over():
            print("Game over")
            break
        print(game.board())
        print()

        move = get_move()
        game.add_main_variation(move)
        game = game.next()
        print(game.board())
        print()


def play_bot_with_bot(sleep_time=5):
    global game
    while not game.board().is_game_over():
        move = get_bot_move()
        if not move:
            print("No legal moves found")
            break
        game.add_main_variation(move)
        game = game.next()
        if game.board().is_game_over():
            print("Game over")
            break
        print(game.board())
        print()
        time.sleep(sleep_time)

        move = get_bot_move()
        if not move:
            print("No legal moves found")
            break
        game.add_main_variation(move)
        game = game.next()
        print(game.board())
        print()
        time.sleep(sleep_time)


play_bot_with_bot()
