"""
LSTM-based forecasting model (simple baseline).

This module provides a thin factory function (`make_model`) that constructs a
`BaseLSTM` instance with configurable hyper-parameters. It exists as a
dedicated entry point so that training scripts and the model registry can
import a consistent `make_model` interface for every architecture (LSTM, TCN,
Transformer) without knowing internal class names.

Usage:
    from chronos.models.lstm_model import make_model
    model = make_model(input_size=1, hidden_size=128, num_layers=3, out_steps=1)
"""

# Import the core LSTM class defined in base.py
from .base import BaseLSTM


def make_model(input_size=1, hidden_size=64, num_layers=2, out_steps=1):
    """
    Factory function that creates and returns a BaseLSTM model.

    This gives callers a uniform `make_model(...)` interface that mirrors the
    factory functions in the TCN and Transformer modules, making it easy to
    swap architectures via configuration without changing calling code.

    Args:
        input_size  (int): Number of input features per timestep (1 for
                           univariate time series).
        hidden_size (int): Width of the LSTM hidden state.  Larger values
                           give the model more capacity at the cost of speed.
        num_layers  (int): Depth of the stacked LSTM.
        out_steps   (int): Number of future steps to predict.

    Returns:
        A `BaseLSTM` instance ready for training or inference.
    """
    return BaseLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, out_steps=out_steps)
