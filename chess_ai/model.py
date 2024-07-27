# https://colab.research.google.com/drive/1SkGW4D3QkNVQkER-OerFXbWJF9DMjd9v?usp=sharing
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ChessAI(nn.Module):
    def __init__(self):
        super(ChessAI, self).__init__()
        self.conv1 = nn.Conv2d(12, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 512, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(512)
        self.conv3 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(1024)
        self.fc1 = nn.Linear(1024 * 2 * 2, 2048)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(2048 + 3, 1024)  # Adding 3 for additional parameters
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(1024, 4864)

    def forward(self, x, params):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(-1, 1024 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.cat(
            (x, params), 1
        )  # Combine board representation with additional parameters
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
