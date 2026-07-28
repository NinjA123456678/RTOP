import torch.nn as nn

class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 1)
        self.norm = nn.LayerNorm(64)

    def forward(self, x):
        h = nn.functional.relu(self.fc1(x))
        h = self.norm(h)
        return self.fc2(h).squeeze()
