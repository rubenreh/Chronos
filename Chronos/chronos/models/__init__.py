"""
Model definitions for Chronos.

This is the package initializer for the `chronos.models` module. It serves as
the public API surface for all forecasting models in the Chronos time-series
prediction system. By importing key classes and factory functions here, the
rest of the application can do:

    from chronos.models import BaseLSTM, make_lstm_model, ...

instead of reaching into individual submodules. This keeps imports clean and
decouples consumers from the internal file layout.

Models exposed:
    - BaseLSTM              – foundational LSTM-based recurrent forecasting model
    - make_lstm_model       – factory that builds a BaseLSTM with sensible defaults
    - TCN / make_tcn_model  – Temporal Convolutional Network for causal conv forecasting
    - TimeSeriesTransformer / make_transformer_model – encoder-only Transformer
    - ModelEnsemble / EnsembleMethod / create_ensemble – combine multiple models
"""

# Import the base LSTM class from the base module
from .base import BaseLSTM

# Import the LSTM factory function (aliased for clarity at the package level)
from .lstm_model import make_model as make_lstm_model

# Import the TCN architecture and its factory function
from .tcn_model import TCN, make_tcn_model

# Import the Transformer architecture and its factory function
from .transformer_model import TimeSeriesTransformer, make_transformer_model

# Import the ensemble wrapper, its strategy enum, and the convenience constructor
from .ensemble import ModelEnsemble, EnsembleMethod, create_ensemble

# __all__ defines the public API when a consumer does `from chronos.models import *`
__all__ = [
    'BaseLSTM',
    'make_lstm_model',
    'TCN',
    'make_tcn_model',
    'TimeSeriesTransformer',
    'make_transformer_model',
    'ModelEnsemble',
    'EnsembleMethod',
    'create_ensemble'
]
