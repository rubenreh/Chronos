"""LSTM-based forecasting model (simple baseline)."""
from .base import BaseLSTM

# This file intentionally small: keep model definition clear and simple.

def make_model(input_size=1, hidden_size=64, num_layers=2, out_steps=1):
    return BaseLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, out_steps=out_steps)
