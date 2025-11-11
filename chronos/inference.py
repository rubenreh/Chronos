"""Inference helpers: load model and run prediction on a sliding window input."""
import torch
import numpy as np
from typing import Union, Tuple
from chronos.models import make_lstm_model

def load_model(path: str, device=None) -> Tuple[torch.nn.Module, float, float]:
    """Load a trained model from checkpoint.
    
    Args:
        path: Path to model checkpoint
        device: Device to load model on (defaults to CPU)
    
    Returns:
        Tuple of (model, mu, sigma) for normalization
    """
    if device is None:
        device = torch.device('cpu')
    
    ckpt = torch.load(path, map_location=device)
    
    # Try to infer model architecture from checkpoint
    if 'model_type' in ckpt:
        model_type = ckpt['model_type']
        if model_type == 'lstm':
            from chronos.models import make_lstm_model
            model = make_lstm_model(
                input_size=1,
                hidden_size=ckpt.get('hidden_size', 64),
                num_layers=ckpt.get('num_layers', 2),
                out_steps=1
            )
        else:
            # Default to LSTM
            model = make_lstm_model(input_size=1, hidden_size=64, num_layers=2, out_steps=1)
    else:
        # Default to LSTM for backward compatibility
        model = make_lstm_model(input_size=1, hidden_size=64, num_layers=2, out_steps=1)
    
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()
    
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
    """Make a prediction from a time-series window.
    
    Args:
        model: Trained PyTorch model
        mu: Mean for normalization
        sigma: Standard deviation for normalization
        window: 1D numpy array of input values
        device: Device to run inference on
    
    Returns:
        Predicted value (denormalized)
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Normalize
    x = (window - mu) / (sigma + 1e-8)
    
    # Convert to tensor
    x_tensor = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-1)  # (1, seq, 1)
    
    # Predict
    with torch.no_grad():
        pred = model(x_tensor)
        if isinstance(pred, torch.Tensor):
            pred = pred.squeeze().cpu().numpy()
        else:
            pred = float(pred)
    
    # Denormalize
    return float(pred * sigma + mu)
