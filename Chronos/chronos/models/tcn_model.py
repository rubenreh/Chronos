"""
Temporal Convolutional Network (TCN) for time-series forecasting.

This module implements a TCN – an architecture built entirely from 1-D causal
convolutions.  Unlike RNNs, a TCN can process an entire sequence in parallel
(no sequential hidden-state dependency), which makes training faster on GPUs.

Key design ideas:
    - **Dilated convolutions**: Each successive layer doubles the dilation
      factor (1, 2, 4, 8, …), so the receptive field grows exponentially
      with depth while keeping the number of parameters linear.
    - **Causal padding + chomp**: Padding is applied only to the left so
      that each output timestep depends only on current and past inputs
      (no future leakage).  A `Chomp1d` layer trims the extra right-side
      padding to enforce this exactly.
    - **Residual connections**: Each `TemporalBlock` adds its input to its
      output (with a 1×1 conv if channel counts differ), helping gradients
      flow through deep networks.

Tensor flow:
    (batch, seq_len, input_size)
      → transpose to (batch, input_size, seq_len) for Conv1d
      → stack of TemporalBlocks (dilated causal conv + residual)
      → take last timestep → (batch, num_channels[-1])
      → linear head         → (batch, output_size)
"""

import torch                   # Core tensor operations
import torch.nn as nn          # Neural-network building blocks
from typing import Optional    # Type hints for optional parameters


class TemporalBlock(nn.Module):
    """
    A single residual block in the TCN.

    Each block contains two dilated causal convolution layers, each followed
    by a chomp (to enforce causality), ReLU activation, and dropout.  A
    residual (skip) connection is added from input to output; if the channel
    dimensions differ, a 1×1 convolution adjusts the input to match.

    Args:
        n_inputs    (int): Number of input channels.
        n_outputs   (int): Number of output channels.
        kernel_size (int): Width of the convolutional kernel.
        stride      (int): Convolution stride (typically 1).
        dilation    (int): Dilation factor – controls how far apart the
                           kernel samples are.  A dilation of 4 means the
                           kernel "skips" 3 elements between each tap.
        padding     (int): Left-side zero-padding = (kernel_size - 1) * dilation.
        dropout   (float): Dropout probability for regularisation.
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2
    ):
        super().__init__()

        # First dilated causal convolution: expands/contracts channel dimension
        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        # Chomp removes extra right-side padding so output length == input length
        self.chomp1 = Chomp1d(padding)
        # ReLU activation introduces non-linearity after the first conv
        self.relu1 = nn.ReLU()
        # Dropout randomly zeroes channels to prevent co-adaptation
        self.dropout1 = nn.Dropout(dropout)

        # Second dilated causal convolution: refines features at the same width
        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Chain the two conv→chomp→relu→dropout sub-blocks into one Sequential
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )

        # If the input and output channel counts differ, use a 1×1 convolution
        # to match dimensions for the residual addition.  If they match, the
        # identity connection is used directly (None).
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

        # Final ReLU applied after the residual addition
        self.relu = nn.ReLU()

        # Initialize convolutional weights with small random values for stable training
        self.init_weights()

    def init_weights(self):
        """
        Initialize convolution weights from a normal distribution with
        std = 0.01. Small initial weights help stabilise early training.
        """
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        Forward pass with residual connection.

        Args:
            x: Tensor of shape (batch, channels, seq_len).

        Returns:
            Tensor of shape (batch, n_outputs, seq_len) after applying
            dilated causal convolutions and adding the residual.
        """
        # Pass input through the two-layer conv sub-network
        out = self.net(x)

        # Compute the residual: identity if channels match, else 1×1 conv
        res = x if self.downsample is None else self.downsample(x)

        # Add the residual to the conv output and apply ReLU.
        # This skip connection eases gradient flow in deep networks.
        return self.relu(out + res)


class Chomp1d(nn.Module):
    """
    Remove trailing elements along the time axis to enforce causal padding.

    When Conv1d is configured with padding = (kernel_size - 1) * dilation,
    the output is longer than the input by `padding` timesteps on the right.
    Chomping those extra timesteps ensures each output position only depends
    on current and earlier input positions (no future information leaks).

    Args:
        chomp_size (int): Number of timesteps to remove from the right end.
    """

    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        Slice off the last `chomp_size` timesteps.

        Args:
            x: Tensor of shape (batch, channels, seq_len + chomp_size).

        Returns:
            Tensor of shape (batch, channels, seq_len) – causally valid.
        """
        # Negative indexing removes the rightmost `chomp_size` elements;
        # .contiguous() ensures the memory layout is contiguous after slicing
        return x[:, :, :-self.chomp_size].contiguous()


class TCN(nn.Module):
    """
    Temporal Convolutional Network for time-series forecasting.

    Stacks multiple `TemporalBlock` layers with exponentially increasing
    dilation factors.  This gives the network an exponentially large
    receptive field with only a linear number of parameters:
        receptive field ≈ 2^num_levels × kernel_size

    For example, 3 layers with kernel_size=3 gives a receptive field of
    ~24 timesteps, and 6 layers covers ~192 timesteps.

    Args:
        input_size   (int):   Features per timestep (1 for univariate).
        output_size  (int):   Prediction dimension.
        num_channels (list):  Channel width for each TemporalBlock layer.
                              Length of this list determines network depth.
        kernel_size  (int):   Width of each convolutional kernel.
        dropout    (float):   Dropout rate for regularisation.
    """

    def __init__(
        self,
        input_size: int = 1,
        output_size: int = 1,
        num_channels: list = [64, 64, 64],
        kernel_size: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        layers = []
        num_levels = len(num_channels)  # Depth of the TCN (number of TemporalBlocks)

        for i in range(num_levels):
            # Dilation doubles at every level: 1, 2, 4, 8, …
            # This is what gives the TCN its exponentially growing receptive field
            dilation_size = 2 ** i

            # First layer takes raw input features; subsequent layers take
            # the output channels of the previous TemporalBlock
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]

            # Build the TemporalBlock with the computed dilation and padding.
            # Padding = (kernel_size - 1) * dilation ensures output length
            # equals input length (before chomping makes it causal).
            layers += [TemporalBlock(
                in_channels, out_channels, kernel_size,
                stride=1, dilation=dilation_size,
                padding=(kernel_size-1) * dilation_size,
                dropout=dropout
            )]

        # Chain all TemporalBlocks into a single Sequential module
        self.network = nn.Sequential(*layers)

        # Linear head: maps the channel vector at the last timestep to predictions
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        """
        Forward pass through the TCN.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size).
               For Chronos this is typically (batch, 60, 1).

        Returns:
            Tensor of shape (batch, output_size) with the forecast.
        """
        # Conv1d expects (batch, channels, seq_len), so swap the last two dims
        x = x.transpose(1, 2)

        # Run through the stack of dilated causal convolution blocks
        y = self.network(x)

        # Take only the output at the LAST timestep – this has the largest
        # receptive field and incorporates information from the full window.
        # Shape: (batch, num_channels[-1])
        y = y[:, :, -1]

        # Project to the final prediction
        return self.linear(y)


def make_tcn_model(
    input_size: int = 1,
    output_size: int = 1,
    num_channels: list = [64, 64, 64],
    kernel_size: int = 3,
    dropout: float = 0.2
) -> TCN:
    """
    Factory function that creates a TCN model.

    Provides the same `make_*_model(...)` interface used by the LSTM and
    Transformer modules, enabling architecture swapping via configuration.

    Args:
        input_size   (int):   Features per timestep.
        output_size  (int):   Prediction dimension.
        num_channels (list):  Channel widths per layer (length = depth).
        kernel_size  (int):   Convolution kernel width.
        dropout    (float):   Regularisation dropout.

    Returns:
        A `TCN` instance ready for training or inference.
    """
    return TCN(
        input_size=input_size,
        output_size=output_size,
        num_channels=num_channels,
        kernel_size=kernel_size,
        dropout=dropout
    )
