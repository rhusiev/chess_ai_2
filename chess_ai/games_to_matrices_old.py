import chess
import chess.pgn
import torch

ELO_RANGES = [
    # 800,
    # 1200,
    1600,
    # 2000,
    # 2400,
]
NUM_FOR_SINGLE_OF_ELO_RANGE = 8000
ELO_RANGE_MUL = [
    # 2,
    # 2,
    4 - 2,
    # 4,
    # 3,
]


def game_to_tensor(game: chess.pgn.GameNode) -> tuple[torch.Tensor, torch.Tensor]:
    # Initialize an 8x8x12 tensor with zeros
    tensor = torch.zeros(8, 8, 12)

    # Mapping of pieces to tensor indices
    piece_to_idx = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5,
    }

    # Current player color
    color = game.turn()

    # Iterate over the board and set the tensor values
    for i in range(64):
        piece = game.board().piece_at(i)
        if piece:
            color_offset = 0 if piece.color == color else 6
            piece_idx = piece_to_idx[piece.piece_type]
            row, col = divmod(i, 8)
            tensor[row, col, color_offset + piece_idx] = 1

    # Add the additional binary features
    queen_castling_right = bool(
        game.board().castling_rights & (chess.BB_A1 if color else chess.BB_A8)
    )
    king_castling_right = bool(
        game.board().castling_rights & (chess.BB_H1 if color else chess.BB_H8)
    )
    consts = torch.tensor(
        [
            queen_castling_right,
            king_castling_right,
            1 if game.turn() == chess.WHITE else 0,
        ]
    )

    return tensor, consts


def move_to_tensor(move: chess.Move) -> torch.Tensor:
    from_sq, to_sq = move.from_square, move.to_square
    from_row, from_col = divmod(from_sq, 8)
    to_row, to_col = divmod(to_sq, 8)

    promotion_tensor = torch.zeros(8, 8, 12)
    queen_moves_tensor = torch.zeros(8, 8, 56)
    knight_moves_tensor = torch.zeros(8, 8, 8)
    if move.promotion:
        if move.promotion == chess.QUEEN:
            promotion_tensor[from_row][from_col][10 + to_col - from_col] = 1
        elif move.promotion == chess.BISHOP:
            promotion_tensor[from_row][from_col][7 + to_col - from_col] = 1
        elif move.promotion == chess.ROOK:
            promotion_tensor[from_row][from_col][4 + to_col - from_col] = 1
        elif move.promotion == chess.KNIGHT:
            promotion_tensor[from_row][from_col][1 + to_col - from_col] = 1
    elif from_row - to_row == from_col - to_col:
        if from_row < to_row:
            queen_moves_tensor[from_row][from_col][
                7 + (abs(from_col - to_col) - 1) * 8
            ] = 1
        else:
            queen_moves_tensor[from_row][from_col][
                3 + (abs(from_col - to_col) - 1) * 8
            ] = 1
    elif from_row - to_row == to_col - from_col:
        if from_row < to_row:
            queen_moves_tensor[from_row][from_col][
                1 + (abs(from_col - to_col) - 1) * 8
            ] = 1
        else:
            queen_moves_tensor[from_row][from_col][
                5 + (abs(from_col - to_col) - 1) * 8
            ] = 1
    elif from_row == to_row:
        if from_col < to_col:
            queen_moves_tensor[from_row][from_col][
                6 + (abs(from_col - to_col) - 1) * 8
            ] = 1
        else:
            queen_moves_tensor[from_row][from_col][
                2 + (abs(from_col - to_col) - 1) * 8
            ] = 1
    elif from_col == to_col:
        if from_row < to_row:
            queen_moves_tensor[from_row][from_col][
                0 + (abs(from_row - to_row) - 1) * 8
            ] = 1
        else:
            queen_moves_tensor[from_row][from_col][
                4 + (abs(from_row - to_row) - 1) * 8
            ] = 1
    elif from_row - to_row == -2:
        if from_col - to_col == -1:
            knight_moves_tensor[from_row][from_col][7] = 1
        else:
            knight_moves_tensor[from_row][from_col][0] = 1
    elif from_row - to_row == 2:
        if from_col - to_col == -1:
            knight_moves_tensor[from_row][from_col][4] = 1
        else:
            knight_moves_tensor[from_row][from_col][3] = 1
    elif from_row - to_row == -1:
        if from_col - to_col == -2:
            knight_moves_tensor[from_row][from_col][6] = 1
        else:
            knight_moves_tensor[from_row][from_col][1] = 1
    else:
        if from_col - to_col == -2:
            knight_moves_tensor[from_row][from_col][5] = 1
        else:
            knight_moves_tensor[from_row][from_col][2] = 1

    one_hot = torch.cat(
        (queen_moves_tensor, knight_moves_tensor, promotion_tensor),
        dim=2,
    )

    return torch.argmax(one_hot)


def process_elos(elo: tuple[int, int], offsets: list, total_games: int) -> None:
    print(f"Processing elos {elo[0]}-{elo[1]}")

    states_tensors, states_consts_tensors, moves_tensors = process_offsets(
        offsets, total_games, pgn
    )
    torch.save(states_tensors, f"./data/20.11/states_tensors_{elo[0]}-{elo[1]}.pt")
    torch.save(
        states_consts_tensors,
        f"./data/20.11/states_consts_tensors_{elo[0]}-{elo[1]}.pt",
    )
    torch.save(moves_tensors, f"./data/20.11/moves_tensors_{elo[0]}-{elo[1]}.pt")

    print("Finished making matrices.")

    print("Finished saving values.")


def get_offsets(pgn, skip=0) -> tuple[dict[int, list[int]], dict[int, int]]:
    print("Finding rated games")
    # The number is the starting elo
    offsets = {}
    counts = {}
    for elo in ELO_RANGES:
        offsets[elo] = []
        counts[elo] = -skip

    count = 0
    print(counts, end="\r")
    while True:
        count += 1
        if sum(counts.values()) >= sum(ELO_RANGE_MUL) * NUM_FOR_SINGLE_OF_ELO_RANGE:
            break
        offset = pgn.tell()
        headers = chess.pgn.read_headers(pgn)
        if not headers:
            break
        elos = headers.get("WhiteElo"), headers.get("BlackElo")
        if elos is None or elos[0] is None or elos[1] is None:
            print("No elos")
            continue
        if int(elos[0]) <= ELO_RANGES[0]:
            continue
        found = False
        for elo in ELO_RANGES[:-1]:
            if int(elos[0]) > elo + 400:
                continue
            found = True
            if (
                counts[elo]
                >= ELO_RANGE_MUL[ELO_RANGES.index(elo)] * NUM_FOR_SINGLE_OF_ELO_RANGE
            ):
                break
            if counts[elo] % 1000 == 0:
                print(counts, end="\r")
            counts[elo] += 1
            if counts[elo] <= 0:
                break
            offsets[elo].append(offset)
            break
        if found:
            continue
        if counts[ELO_RANGES[-1]] >= ELO_RANGE_MUL[-1] * NUM_FOR_SINGLE_OF_ELO_RANGE:
            continue
        if counts[ELO_RANGES[-1]] % 1000 == 0:
            print(counts, end="\r")
        counts[ELO_RANGES[-1]] += 1
        if counts[ELO_RANGES[-1]] <= 0:
            continue
        offsets[ELO_RANGES[-1]].append(offset)
    print(counts)
    print("Found all games required")
    return offsets, counts


def process_offsets(offsets: list, total_games: int, pgn):
    states_tensors = []
    states_consts_tensors = []
    moves_tensors = []

    count = 0
    print("Finished 0% of this rated games", end="\r")
    for offset in offsets:
        pgn.seek(offset)
        game = chess.pgn.read_game(pgn)
        if not game:
            continue
        print(
            f"Finished {(count / total_games * 100):.2f}% of this rated games", end="\r"
        )
        if count > total_games:
            break
        if game.errors:
            continue
        game = game.next()
        if not game:
            continue
        state_tensor, state_consts_tensor = game_to_tensor(game)
        while game := game.next():
            move_tensor = move_to_tensor(game.move)
            states_tensors.append(state_tensor)
            states_consts_tensors.append(state_consts_tensor)
            moves_tensors.append(move_tensor)
            state_tensor, state_consts_tensor = game_to_tensor(game)
        count += 1
    print("Finished 100% of this rated games")
    return (
        torch.stack(states_tensors),
        torch.stack(states_consts_tensors),
        torch.stack(moves_tensors),
    )


if __name__ == "__main__":
    # path = "../test/neural_chess/lichess_2022_02.pgn"
    path = "./data/lichess_2020-11.pgn"
    with open(path) as pgn:
        offsets, counts = get_offsets(pgn)
        # for i in range(800, 2401, 400):
            # process_elos((i, i + 400), offsets[i], counts[i])
        i = 1600
        process_elos((i, i + 400), offsets[i], counts[i])
