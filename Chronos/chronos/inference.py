"""
Inference Module – Model Loading and Prediction for Chronos
=============================================================
This module provides the inference-time entry points for the Chronos system:

  • load_model  – deserializes a saved checkpoint (.pth), reconstructs the
                  correct model architecture (LSTM by default), loads the
                  trained weights, and returns the model along with the
                  normalization parameters (mu, sigma) that were stored at
                  training time.
  • predict     – takes a trained model and a raw time-series window, applies
                  z-score normalization (using mu/sigma from training), runs
                  the forward pass, and denormalizes the output back to the
                  original value scale.

The normalize → predict → denormalize pipeline is critical: the model was
trained on z-score-normalized data, so it must receive normalized input and
its output must be scaled back to real-world units before being returned to
the caller (e.g., the FastAPI endpoint).
"""

import torch
import numpy as np
from typing import Union, Tuple

# Default model factory — used when the checkpoint doesn't specify a type.
from chronos.models import make_lstm_model


def load_model(path: str, device=None) -> Tuple[torch.nn.Module, float, float]:
    """Load a trained model from a checkpoint file.

    The checkpoint is expected to contain at minimum:
        - 'model_state'  : the state_dict of the trained model
        - 'mu'           : mean used for z-score normalization during training
        - 'sigma'        : std dev used for z-score normalization during training
    Optionally:
        - 'model_type'   : string identifying the architecture ('lstm', 'tcn', etc.)
        - 'hidden_size'  : LSTM hidden dimension (defaults to 64)
        - 'num_layers'   : number of stacked LSTM layers (defaults to 2)

    Args:
        path: Filesystem path to the .pth checkpoint file.
        device: torch.device to map the model onto. Defaults to CPU, which is
                safest for inference (works everywhere, including Docker containers
                without GPU).

    Returns:
        Tuple of (model, mu, sigma):
            model – the reconstructed nn.Module with trained weights loaded.
            mu    – the training-set mean for normalizing new input data.
            sigma – the training-set std dev for normalizing new input data.
    """
    # Default to CPU for broad compatibility during inference.
    if device is None:
        device = torch.device('cpu')

    # Load the checkpoint, mapping tensors to the specified device.
    ckpt = torch.load(path, map_location=device)

    # Determine which architecture to build based on the checkpoint metadata.
    if 'model_type' in ckpt:
        model_type = ckpt['model_type']

        if model_type == 'lstm':
            # Dynamically import to avoid circular imports at module level.
            from chronos.models import make_lstm_model
            # Reconstruct the LSTM with the same hyperparameters used in training.
            model = make_lstm_model(
                input_size=1,
                hidden_size=ckpt.get('hidden_size', 64),
                num_layers=ckpt.get('num_layers', 2),
                out_steps=1
            )
        else:
            # Fallback to default LSTM if model_type is unrecognized.
            model = make_lstm_model(input_size=1, hidden_size=64, num_layers=2, out_steps=1)
    else:
        # Legacy checkpoints without model_type default to LSTM for backward compatibility.
        model = make_lstm_model(input_size=1, hidden_size=64, num_layers=2, out_steps=1)

    # Load the trained weights into the model architecture.
    model.load_state_dict(ckpt['model_state'])

    # Move model to the target device.
    model.to(device)

    # Set model to evaluation mode (disables dropout, freezes batch-norm).
    model.eval()

    # Extract normalization parameters; default to identity transform (mu=0, sigma=1)
    # if they're missing, so predictions are returned as-is.
    mu = ckpt.get('mu', 0.0)
    sigma = ckpt.get('sigma', 1.0)

    return model, mu, sigma


def predict(
    model: torch.nn.Module,
    mu: float,
    sigma: float,
    window: np.ndarray,
    device: torch.device = None
) -> float:
    """Make a single-step prediction from a raw time-series window.

    The function follows the normalize → predict → denormalize pipeline:
      1. Apply z-score normalization to the input window using the training-set
         mu and sigma.
      2. Convert the normalized window to a PyTorch tensor with shape (1, seq_len, 1).
      3. Run the forward pass through the model (no gradients needed).
      4. Denormalize the model output back to the original value scale.

    Args:
        model: Trained PyTorch model (already in eval mode).
        mu: Training-set mean for normalization.
        sigma: Training-set std dev for normalization.
        window: 1-D numpy array of raw (un-normalized) input values.
                Length should match the input_len the model was trained with.
        device: Device to run inference on. If None, uses the device the model
                is currently on.

    Returns:
        A single float: the predicted next value, denormalized to the original scale.
    """
    # Determine the device from the model's parameters if not explicitly given.
    if device is None:
        device = next(model.parameters()).device

    # Step 1: Z-score normalize the raw input window.
    # Adding 1e-8 to sigma prevents division by zero if the training data had
    # zero variance (unlikely but defensive).
    x = (window - mu) / (sigma + 1e-8)

    # Step 2: Convert to a PyTorch tensor and reshape.
    # unsqueeze(0) adds a batch dimension:       (seq_len,) → (1, seq_len)
    # unsqueeze(-1) adds a feature dimension:    (1, seq_len) → (1, seq_len, 1)
    # The model expects input shape (batch, seq_len, n_features).
    x_tensor = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-1)  # (1, seq, 1)

    # Step 3: Forward pass with gradient tracking disabled (faster, less memory).
    with torch.no_grad():
        pred = model(x_tensor)

        # Handle different output types: squeeze any extra dimensions and
        # move from GPU to CPU, converting to a numpy scalar.
        if isinstance(pred, torch.Tensor):
            pred = pred.squeeze().cpu().numpy()
        else:
            pred = float(pred)

    # Step 4: Denormalize — reverse the z-score transform to get the
    # prediction in the original value units (e.g., temperature in °F,
    # stock price in $, sensor reading in mV).
    return float(pred * sigma + mu)
