"""Base model utilities for Chronos (PyTorch helpers)."""

import torch
import torch.nn as nn


class BaseLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.1, out_steps=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, out_steps)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, (h, c) = self.lstm(x)
        # take last timestep
        last = out[:, -1, :]
        return self.fc(last)
