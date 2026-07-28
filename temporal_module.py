import torch
import torch.nn as nn

class TemporalModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv1d(1, 16, 5, padding=2)
        self.attn = nn.MultiheadAttention(32, 4, batch_first=True)

    def forward(self, x):
        x = x.unsqueeze(1)
        h1 = torch.relu(self.conv1(x))
        h2 = torch.relu(self.conv2(x))
        h = torch.cat([h1, h2], dim=1)
        h = h.permute(0, 2, 1)
        out, _ = self.attn(h, h, h)
        return out.mean(dim=1)