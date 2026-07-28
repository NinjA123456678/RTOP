import torch
import torch.nn as nn

class SDAE(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        mask = torch.bernoulli(torch.full_like(x, 0.1))
        x_noisy = x * (1 - mask)

        z = self.encoder(x_noisy)
        recon = self.decoder(z)

        error = (x - recon) ** 2
        error = error.mean(dim=1)

        s = (error - error.min()) / (error.max() - error.min() + 1e-8)
        w = 1 - s

        return recon, w, error.mean()