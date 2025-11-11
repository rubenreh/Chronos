"""Simple sequence autoencoder for anomaly detection (PyTorch)."""
import torch
import torch.nn as nn


class SeqAutoencoder(nn.Module):
    def __init__(self, seq_len=60, hidden=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(seq_len, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden//2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden//2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, seq_len)
        )

    def forward(self, x):
        # x expected shape (batch, seq_len)
        z = self.encoder(x)
        return self.decoder(z)
