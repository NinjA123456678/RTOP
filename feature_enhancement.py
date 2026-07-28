import torch
import torch.nn as nn

class FeatureEnhancement(nn.Module):
    def __init__(self, input_dim=64):
        super().__init__()

        # 多头注意力
        self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=4, batch_first=True)

        # 卷积层
        self.conv1 = nn.Conv1d(input_dim, input_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(input_dim, input_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(input_dim, input_dim, kernel_size=3, padding=1)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        # x: (B, D)
        x = x.unsqueeze(1)  # (B,1,D)

        # attention 需要 (B, T, D)
        attn_out, _ = self.attn(x, x, x)

        # conv 需要 (B, D, T)
        h = attn_out.permute(0, 2, 1)

        h = torch.relu(self.conv1(h))
        h = torch.relu(self.conv2(h))
        h = torch.relu(self.conv3(h))

        h = self.pool(h).squeeze(-1)
        h = self.norm(h)

        return h