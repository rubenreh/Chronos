"""
Transformer model for time-series forecasting.

This module implements an encoder-only Transformer tailored for sequential
prediction tasks in the Chronos system. Unlike the LSTM which processes
tokens one-by-one, the Transformer uses self-attention to relate every
timestep to every other timestep in parallel, capturing long-range
dependencies more efficiently.

Architecture overview:
    1. Input Projection – linear layer maps raw features to d_model dimensions
    2. Positional Encoding – injects sine/cosine position signals so the
       model knows the ordering of timesteps (Transformers have no built-in
       notion of sequence order)
    3. Transformer Encoder – stack of multi-head self-attention + feed-forward
       sublayers that learn temporal relationships
    4. Output Projection – linear layer maps the final-timestep encoding
       down to the prediction dimension

Tensor flow:
    (batch, seq_len, input_size)
      → transpose to (seq_len, batch, input_size) for PyTorch Transformer
      → input projection  → (seq_len, batch, d_model)
      → positional encoding
      → N encoder layers
      → select last timestep → (batch, d_model)
      → output projection   → (batch, output_size)
"""

import torch            # Core tensor operations
import torch.nn as nn   # Neural-network layers
import math             # For sqrt and log used in scaling / positional encoding


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding (from "Attention Is All You Need").

    Transformers process all timesteps simultaneously so they have no
    inherent sense of order.  This module adds a unique, deterministic
    signal to each position so the model can distinguish timestep 0 from
    timestep 59.  The encoding uses sine for even dimensions and cosine
    for odd dimensions at exponentially increasing frequencies, giving each
    position a unique "fingerprint" that the attention layers can learn to
    exploit.

    Args:
        d_model (int): Embedding dimension (must match the rest of the
                       Transformer so the addition is valid).
        max_len (int): Maximum sequence length the encoding supports.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        # pe: (max_len, d_model) – one row per position, one column per dim
        pe = torch.zeros(max_len, d_model)

        # position: column vector [0, 1, 2, ..., max_len-1]
        # Each entry is the absolute position index in the sequence
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # div_term: controls the frequency for each dimension pair.
        # Frequencies decrease exponentially from high (dim 0) to low (dim d_model).
        # Formula: exp(2i * (-ln(10000) / d_model)) = 1 / 10000^(2i/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Even dimensions (0, 2, 4, …) use sine
        pe[:, 0::2] = torch.sin(position * div_term)
        # Odd dimensions (1, 3, 5, …) use cosine
        pe[:, 1::2] = torch.cos(position * div_term)

        # Reshape to (max_len, 1, d_model) so it broadcasts over the batch dim
        pe = pe.unsqueeze(0).transpose(0, 1)

        # Register as a buffer (saved with the model but not a learnable parameter)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Add positional encoding to the input embeddings.

        Args:
            x: Tensor of shape (seq_len, batch, d_model) – the projected
               input embeddings before entering the Transformer encoder.

        Returns:
            Tensor of the same shape with positional information added
            element-wise to each embedding vector.
        """
        # Slice pe to match the actual sequence length and add to x
        return x + self.pe[:x.size(0), :]


class TimeSeriesTransformer(nn.Module):
    """
    Encoder-only Transformer for time-series forecasting.

    This model projects raw input features into a higher-dimensional space
    (d_model), adds positional encoding, passes the result through a stack
    of Transformer encoder layers (self-attention + feed-forward), and
    finally maps the last timestep's representation to the prediction.

    Args:
        input_size      (int): Number of input features per timestep.
        d_model         (int): Internal embedding dimension used throughout
                               the Transformer.  Must be divisible by nhead.
        nhead           (int): Number of attention heads.  Each head attends
                               to a d_model/nhead dimensional subspace.
        num_layers      (int): Number of stacked encoder layers.
        dim_feedforward (int): Width of the position-wise feed-forward
                               network inside each encoder layer.
        dropout       (float): Dropout rate for regularisation.
        output_size     (int): Number of values to predict (1 for single-step).
    """

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

        # Linear projection from raw feature space to d_model dimensions.
        # This lets the model operate in a richer representation space.
        self.input_projection = nn.Linear(input_size, d_model)

        # Add sinusoidal positional encoding so the model knows timestep order
        self.pos_encoder = PositionalEncoding(d_model)

        # Define a single encoder layer (self-attention + feed-forward + norms)
        # batch_first=False means tensors are (seq_len, batch, d_model)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )

        # Stack `num_layers` copies of the encoder layer
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # Final linear head: maps the d_model representation at the last
        # timestep to the number of values we want to predict
        self.output_projection = nn.Linear(d_model, output_size)

        # Store d_model for the scaling factor applied after input projection
        self.d_model = d_model

    def forward(self, x):
        """
        Forward pass through the Transformer encoder.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size).
               For Chronos this is typically (batch, 60, 1).

        Returns:
            Tensor of shape (batch, output_size) with the forecast.
        """
        # Transpose to (seq_len, batch, input_size) because PyTorch's
        # TransformerEncoder expects sequence-first when batch_first=False
        x = x.transpose(0, 1)

        # Project features to d_model dims and scale by sqrt(d_model).
        # The scaling prevents the dot-product attention values from growing
        # too large in magnitude, stabilising gradients (same trick used in
        # the original "Attention Is All You Need" paper).
        x = self.input_projection(x) * math.sqrt(self.d_model)

        # Inject positional information into the embeddings
        x = self.pos_encoder(x)

        # Run through the stack of Transformer encoder layers.
        # Each layer applies multi-head self-attention followed by a
        # position-wise feed-forward network (with residual connections
        # and layer normalisation).
        x = self.transformer_encoder(x)

        # Extract the representation at the LAST timestep, which has
        # attended to all preceding positions. Shape: (batch, d_model)
        x = x[-1]

        # Map the d_model vector to the final prediction(s)
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
    """
    Factory function that creates a TimeSeriesTransformer.

    Provides the same `make_*_model(...)` interface used by the LSTM and TCN
    modules so that training scripts can swap architectures via configuration.

    Args:
        input_size      (int): Features per timestep.
        d_model         (int): Transformer embedding width.
        nhead           (int): Number of attention heads.
        num_layers      (int): Encoder depth.
        dim_feedforward (int): Feed-forward hidden width.
        dropout       (float): Regularisation dropout rate.
        output_size     (int): Prediction dimension.

    Returns:
        A `TimeSeriesTransformer` instance ready for training or inference.
    """
    return TimeSeriesTransformer(
        input_size=input_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        output_size=output_size
    )
