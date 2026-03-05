"""
Base model utilities for Chronos (PyTorch helpers).

This module defines `BaseLSTM`, the foundational recurrent neural network model
used throughout the Chronos forecasting system. It wraps PyTorch's LSTM with a
fully connected (linear) output head so it can ingest a sliding window of
time-series observations and produce one or more future-step predictions.

Architecture overview:
    Input  → (batch, seq_len, input_size)   e.g. (32, 60, 1)
    LSTM   → processes the sequence, outputs hidden states for every timestep
    Select → take the hidden state at the LAST timestep (captures full context)
    Linear → project that hidden state to the desired number of output steps

This model is intentionally simple so it can serve as:
    1. A standalone baseline forecaster
    2. One member of a `ModelEnsemble` alongside TCN / Transformer models
"""

import torch            # Core tensor library
import torch.nn as nn   # Neural-network building blocks (layers, loss, etc.)


class BaseLSTM(nn.Module):
    """
    LSTM-based time-series forecasting model.

    This is the simplest deep-learning forecaster in Chronos. It feeds a
    sequence of past observations through stacked LSTM layers and maps the
    final hidden state to the predicted future value(s) via a linear layer.

    Args:
        input_size  (int): Number of features per timestep (1 for univariate).
        hidden_size (int): Dimensionality of each LSTM layer's hidden state.
                           Larger → more capacity but slower to train.
        num_layers  (int): How many LSTM layers are stacked.  Stacking lets
                           the network learn increasingly abstract temporal
                           representations.
        dropout   (float): Dropout probability applied between LSTM layers
                           (not after the last layer) to reduce overfitting.
        out_steps   (int): Number of future timesteps to predict (1 = single-
                           step forecast).
    """

    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.1, out_steps=1):
        super().__init__()

        # nn.LSTM: the core recurrent layer.
        #   - input_size:   features per timestep fed into the first LSTM layer
        #   - hidden_size:  width of the hidden / cell state vectors
        #   - num_layers:   number of stacked LSTM layers (deep LSTM)
        #   - batch_first:  True means input/output tensors are (batch, seq, feature)
        #   - dropout:      applied between layers (except the last) for regularisation
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

        # Fully connected head: maps the last hidden state (hidden_size) to
        # the desired number of output predictions (out_steps).
        self.fc = nn.Linear(hidden_size, out_steps)

    def forward(self, x):
        """
        Forward pass: run the input sequence through the LSTM and project the
        final hidden state to a prediction.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size).
               For Chronos, this is typically (batch, 60, 1) — a window of 60
               past productivity values.

        Returns:
            Tensor of shape (batch, out_steps) with the forecasted value(s).
        """
        # Pass the full sequence through the LSTM.
        # `out`  – hidden states for every timestep: (batch, seq_len, hidden_size)
        # `h`    – final hidden state per layer:     (num_layers, batch, hidden_size)
        # `c`    – final cell state per layer:        (num_layers, batch, hidden_size)
        out, (h, c) = self.lstm(x)

        # Select the hidden state at the LAST timestep. This vector has "seen"
        # the entire input window and encodes the temporal context needed for
        # forecasting. Shape: (batch, hidden_size)
        last = out[:, -1, :]

        # Project the last hidden state to the output dimension.
        # Shape: (batch, out_steps)
        return self.fc(last)
