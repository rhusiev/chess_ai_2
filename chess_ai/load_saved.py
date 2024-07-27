import torch

from chess_ai.model import ChessAI
from chess_ai.model_initialize import optimizer

def load_model(path: str):
    # Load saved model
    model = ChessAI()
    # Load saved state dict
    model.load_state_dict(torch.load(path))
    return model

def load_model_and_optimizer(path: str):
    # Assuming the optimizer is already defined with the same parameters
    model = ChessAI()
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return model, optimizer
