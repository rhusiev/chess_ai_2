import torch
from torch.utils.data import DataLoader, Dataset

from chess_ai.model_initialize import model

DIR = "drive/MyDrive/Colab Notebooks"
DATA_DIR = f"{DIR}/tensors"
SAVE_DIR = f"{DIR}/chess_ai"

states_800_1200 = torch.load(f"{DATA_DIR}/states_tensors_800-1200.pt")
print(f"{states_800_1200.size() = }")
states_consts_800_1200 = torch.load(f"{DATA_DIR}/states_consts_tensors_800-1200.pt")
print(f"{states_consts_800_1200.size() = }")
moves_800_1200 = torch.load(f"{DATA_DIR}/moves_tensors_800-1200.pt")
print(f"{moves_800_1200.size() = }")


class ChessDataset(Dataset):
    def __init__(self, states, states_consts, moves):
        self.states = states
        self.states_consts = states_consts
        self.moves = moves

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        # one_hot_move = torch.zeros(4864)
        # one_hot_move[self.moves[idx]] = 1
        state = self.states[idx].permute(2, 0, 1)  # Change to [12, 8, 8]
        state_const = self.states_consts[idx]
        move = self.moves[idx]
        return state, state_const, move


# Create dataset
chess_dataset = ChessDataset(states_800_1200, states_consts_800_1200, moves_800_1200)

# DataLoader
dataloader = DataLoader(chess_dataset, batch_size=32, shuffle=True, drop_last=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(device)
