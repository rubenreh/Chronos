"""
Simple sequence autoencoder for anomaly detection (PyTorch).

This module implements `SeqAutoencoder`, an autoencoder neural network used
by Chronos to detect anomalous patterns in time-series data (e.g. unusual
productivity spikes or drops).

How anomaly detection works:
    1. TRAINING: The autoencoder is trained on NORMAL sequences only.  It
       learns to compress a sequence into a small latent vector (encoder)
       and then reconstruct the original sequence from that vector (decoder).
    2. INFERENCE: Given a new sequence, the model tries to reconstruct it.
       If the reconstruction error (e.g. MSE) is LOW, the input looks like
       the normal data it was trained on → not anomalous.
       If the error is HIGH, the input deviates from learned patterns →
       flagged as an anomaly.

Architecture:
    Encoder: seq_len → hidden → hidden/2  (compress)
    Decoder: hidden/2 → hidden → seq_len  (reconstruct)

    The bottleneck (hidden/2) forces the network to learn the most
    important features of normal sequences, discarding noise.

Tensor flow:
    Input  (batch, seq_len)   e.g. (32, 60)
    Encoder → (batch, hidden/2)   compressed latent representation
    Decoder → (batch, seq_len)    reconstructed sequence
    Loss = MSE(input, reconstructed)
"""

import torch            # Core tensor library
import torch.nn as nn   # Neural-network layers


class SeqAutoencoder(nn.Module):
    """
    Sequence autoencoder for anomaly detection.

    A fully connected (MLP) autoencoder that takes a flattened time-series
    window and reconstructs it.  During inference, the reconstruction
    error serves as an anomaly score: high error → probable anomaly.

    Note: The input is expected to be a flat 1-D vector per sample
    (batch, seq_len) rather than (batch, seq_len, features).  For
    univariate series this is natural; for multivariate you would flatten
    first.

    Args:
        seq_len (int): Length of each input sequence (default 60, matching
                       the Chronos sliding-window size).
        hidden  (int): Width of the first hidden layer.  The bottleneck
                       (latent dimension) is hidden // 2, so a hidden of
                       64 gives a 32-dimensional latent space.
    """

    def __init__(self, seq_len=60, hidden=64):
        super().__init__()

        # ── Encoder ──
        # Compresses the input sequence down to a compact latent vector.
        self.encoder = nn.Sequential(
            # First layer: reduce from seq_len dimensions to `hidden`
            nn.Linear(seq_len, hidden),
            # ReLU activation: introduces non-linearity so the network can
            # learn complex patterns beyond simple linear mappings
            nn.ReLU(),
            # Second layer: further compress to the bottleneck (hidden // 2)
            nn.Linear(hidden, hidden//2),
            # Another ReLU before the latent representation
            nn.ReLU()
        )

        # ── Decoder ──
        # Reconstructs the original sequence from the latent vector.
        # Mirrors the encoder architecture in reverse.
        self.decoder = nn.Sequential(
            # First layer: expand from bottleneck back to `hidden`
            nn.Linear(hidden//2, hidden),
            # ReLU activation in the reconstruction path
            nn.ReLU(),
            # Final layer: map back to the original sequence length.
            # No activation here – the output is a real-valued reconstruction
            # that should match the raw input values.
            nn.Linear(hidden, seq_len)
        )

    def forward(self, x):
        """
        Forward pass: encode the input to a latent vector, then decode it
        back to the original dimensionality.

        Args:
            x: Input tensor of shape (batch, seq_len).  Each row is one
               flattened time-series window (e.g. 60 productivity values).

        Returns:
            Reconstructed tensor of shape (batch, seq_len).  Compare this
            to the original input to compute reconstruction error (the
            anomaly score).
        """
        # Compress the input sequence to a latent representation
        # Shape: (batch, seq_len) → (batch, hidden//2)
        z = self.encoder(x)

        # Reconstruct the sequence from the latent representation
        # Shape: (batch, hidden//2) → (batch, seq_len)
        return self.decoder(z)
