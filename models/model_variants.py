import torch
import torch.nn as nn


class BP_AttentionBiLSTM(nn.Module):
    """BiLSTM with multi-head self-attention and residual connection."""

    def __init__(self, num_heads=8, attn_dropout=0.1):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32),
        )

        self.rnn = nn.LSTM(128, 128, batch_first=True, bidirectional=True)

        self.attention = nn.MultiheadAttention(
            embed_dim=256, num_heads=num_heads,
            dropout=attn_dropout, batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(256)

        self.regressor = nn.Sequential(
            nn.Linear(256 * 32, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.transpose(1, 2)
        x, _ = self.rnn(x)
        attn_out, _ = self.attention(x, x, x)
        x = self.attn_norm(x + attn_out)
        x = x.reshape(x.size(0), -1)
        return self.regressor(x)


class ResidualTemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, dilation=1, dropout=0.1):
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        residual = self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)
        out = self.dropout(out)
        return out + residual


class BP_TCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(128),
        )

        self.blocks = nn.Sequential(
            ResidualTemporalBlock(64, 64, dilation=1, dropout=0.1),
            ResidualTemporalBlock(64, 64, dilation=2, dropout=0.1),
            ResidualTemporalBlock(64, 128, dilation=4, dropout=0.1),
            ResidualTemporalBlock(128, 128, dilation=8, dropout=0.1),
            ResidualTemporalBlock(128, 128, dilation=16, dropout=0.1),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)


class BP_HybridTransformer(nn.Module):
    def __init__(self, token_len=64, d_model=128, nhead=8, num_layers=3, dropout=0.2):
        super().__init__()
        self.token_len = token_len

        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, d_model, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(token_len),
        )

        self.pos_emb = nn.Parameter(torch.zeros(1, token_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.stem(x)
        x = x.transpose(1, 2)
        x = x + self.pos_emb
        x = self.encoder(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)
