"""Transformer model for time-series forecasting."""
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TimeSeriesTransformer(nn.Module):
    """Transformer for time-series forecasting."""
    
    def __init__(
        self,
        input_size: int = 1,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        output_size: int = 1
    ):
        super().__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.output_projection = nn.Linear(d_model, output_size)
        self.d_model = d_model
    
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = x.transpose(0, 1)  # (seq_len, batch, input_size)
        x = self.input_projection(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        # Take last timestep
        x = x[-1]  # (batch, d_model)
        return self.output_projection(x)


def make_transformer_model(
    input_size: int = 1,
    d_model: int = 128,
    nhead: int = 8,
    num_layers: int = 4,
    dim_feedforward: int = 512,
    dropout: float = 0.1,
    output_size: int = 1
) -> TimeSeriesTransformer:
    """Create a transformer model."""
    return TimeSeriesTransformer(
        input_size=input_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        output_size=output_size
    )

