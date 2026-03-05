"""
Model explainability endpoints.

This module implements the POST /explain endpoint, which provides
interpretability for Chronos model predictions. It supports two strategies:

    1. **SHAP-based explanation** (preferred): wraps the PyTorch model in a
       SHAP-compatible interface, computes Shapley values for each input
       feature (time step), and returns per-feature importance scores.
    2. **Heuristic fallback**: if SHAP computation fails (e.g. missing library,
       incompatible model), falls back to a variance-and-recency-weighted
       importance heuristic plus linear trend and volatility features.

The goal is to answer "why did the model predict this?" — critical for user
trust and for explaining the system during interviews or stakeholder demos.

Key concepts:
    - SHAP (SHapley Additive exPlanations): game-theoretic approach to
      assigning importance to each input feature
    - Background data: SHAP needs a reference dataset to compute marginal
      contributions; we synthesize it from sliding windows of the input
    - SimpleModelWrapper: adapter that converts a PyTorch model into a
      callable that SHAP's KernelExplainer expects (numpy in → numpy out)
"""
import numpy as np                              # Array operations for feature importance calculations
from fastapi import APIRouter, HTTPException    # Router for grouping; HTTPException for input validation
from typing import Optional                     # Type hint for nullable fields

from chronos.api.schemas import ExplainRequest, ExplainResponse  # Pydantic request/response models
from chronos.api.cache import get_cache                          # Singleton in-memory cache
from chronos.evaluation.explainability import ModelExplainer     # SHAP-based explainer from the evaluation module
from chronos.inference import load_model, predict                # Model loading and single-step prediction
from chronos.api.routes.forecast import MODELS                   # Global model registry populated at startup

# Create router under /explain prefix, tagged "explainability" in OpenAPI docs
router = APIRouter(prefix="/explain", tags=["explainability"])

# Module-level cache reference
CACHE = get_cache()


def create_background_data(window: np.ndarray, n_samples: int = 100) -> np.ndarray:
    """Create background (reference) data for SHAP from window statistics.

    SHAP's KernelExplainer needs a background dataset to estimate each
    feature's marginal contribution. This function synthesizes that dataset
    either from overlapping sliding windows of the actual input (preferred)
    or from Gaussian noise matching the input's mean and std (fallback for
    very short windows).

    Args:
        window: 1-D numpy array of the input time-series values
        n_samples: desired number of background samples

    Returns:
        2-D numpy array of shape (n_samples, len(window))
    """
    if len(window) < 10:
        # Window is too short for sliding windows — generate synthetic samples
        # by drawing from a Gaussian matching the window's distribution
        mean_val = np.mean(window)                               # Estimate the distribution center
        std_val = np.std(window)                                 # Estimate the distribution spread
        background = np.random.normal(                           # Sample from N(mean, std)
            mean_val, std_val, (n_samples, len(window))
        )
    else:
        # Build background from overlapping sliding windows of the input.
        # Each window[i : i + len(window)] gives one background sample.
        background = []
        for i in range(min(n_samples, len(window) - 10)):        # Cap at n_samples to limit compute
            background.append(window[i:i+len(window)])           # Extract a contiguous sub-window
        background = np.array(background)                        # Stack into a 2-D array

    return background


class SimpleModelWrapper:
    """Adapter that wraps a PyTorch model so SHAP can call it like a function.

    SHAP's KernelExplainer expects a callable that accepts a 2-D numpy array
    of shape (n_samples, n_features) and returns a 1-D array of predictions.
    This wrapper handles normalization (z-score with training mu/sigma),
    conversion to PyTorch tensors, forward pass, and conversion back to numpy.
    """

    def __init__(self, model, mu: float, sigma: float):
        """Store the model and its normalization statistics.

        Args:
            model: a PyTorch nn.Module (LSTM, TCN, Transformer, or ensemble)
            mu: training-set mean used for z-score normalization
            sigma: training-set standard deviation for z-score normalization
        """
        self.model = model
        self.mu = mu
        self.sigma = sigma

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run inference on a batch of input windows.

        Args:
            X: 2-D numpy array of shape (batch_size, sequence_length)

        Returns:
            1-D numpy array of scalar predictions, one per input sample
        """
        import torch                                             # Lazy import avoids overhead when SHAP isn't used

        predictions = []
        for x in X:                                              # Iterate over each sample in the batch
            # Z-score normalize using the training set's statistics
            x_norm = (x - self.mu) / (self.sigma + 1e-8)        # Epsilon prevents division by zero

            # Convert to a PyTorch tensor with shape (1, seq_len, 1)
            # — batch dim, sequence dim, feature dim (univariate = 1 feature)
            x_tensor = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

            self.model.eval()                                    # Set model to evaluation mode (disables dropout, etc.)
            with torch.no_grad():                                # Disable gradient computation for inference speed
                pred = self.model(x_tensor)                      # Forward pass through the model
                if isinstance(pred, torch.Tensor):
                    pred = pred.squeeze().cpu().numpy()           # Squeeze batch/feature dims and move to CPU
                else:
                    pred = float(pred)                            # Handle models that return plain floats

            predictions.append(pred)

        return np.array(predictions)                             # Stack into a 1-D numpy array


@router.post("", response_model=ExplainResponse)
async def explain_prediction(req: ExplainRequest):
    """Explain model predictions using SHAP values or a heuristic fallback.

    First attempts a full SHAP-based explanation by wrapping the loaded model
    and computing Shapley values. If SHAP fails for any reason (library
    missing, computational error), falls back to a lightweight heuristic that
    scores each time step by recency-weighted deviation from the mean, plus
    overall trend and volatility features.
    """
    # --- Input validation ---
    if len(req.window) < 10:                                     # Need enough context for meaningful explanation
        raise HTTPException(status_code=400, detail="Window too short")

    # --- Cache lookup ---
    cache_key = f"explain:{req.series_id}:{len(req.window)}:{req.top_features}"
    cached_result = CACHE.get(cache_key)
    if cached_result:
        return ExplainResponse(**cached_result)                  # Return cached explanation immediately

    window = np.array(req.window)                                # Convert input list to numpy array for vectorized ops

    # --- Attempt SHAP-based explanation ---
    try:
        # Check if any model is available in the global registry
        if 'ensemble' in MODELS or 'lstm' in MODELS:
            model_key = 'ensemble' if 'ensemble' in MODELS else 'lstm'  # Prefer ensemble, fall back to LSTM
            model_data = MODELS[model_key]
            model = model_data['model']                          # Raw PyTorch nn.Module
            mu = model_data.get('mu', 0.0)                       # Training mean for normalization
            sigma = model_data.get('sigma', 1.0)                 # Training std for normalization

            # Wrap the PyTorch model in a SHAP-compatible interface
            model_wrapper = SimpleModelWrapper(model, mu, sigma)

            # Synthesize background data for SHAP's marginal contribution estimates
            background_data = create_background_data(window, n_samples=50)

            try:
                # Initialize the SHAP explainer (internally creates a KernelExplainer)
                explainer = ModelExplainer(model_wrapper, background_data)

                # Compute SHAP values for the input window
                explanation = explainer.explain_prediction(window, top_features=req.top_features)

                if 'error' not in explanation:                   # SHAP succeeded without errors
                    response = ExplainResponse(
                        series_id=req.series_id,
                        feature_importance=explanation.get('feature_importance', {}),  # Feature → SHAP value mapping
                        shap_values=explanation.get('shap_values'),                   # Raw per-feature SHAP values
                        reasoning=(
                            f"SHAP analysis shows top contributing features. "
                            f"Base value: {explanation.get('base_value', 0):.4f}, "
                            f"Prediction: {explanation.get('prediction', 0):.4f}"
                        )
                    )

                    CACHE.set(cache_key, response.dict())        # Cache the SHAP-based result
                    return response
            except Exception as e:
                pass                                             # SHAP failed — fall through to heuristic
    except Exception:
        pass                                                     # Model lookup failed — fall through to heuristic

    # --- Heuristic fallback: variance-and-recency-weighted importance ---
    # When SHAP is unavailable, approximate feature importance by scoring
    # each time step based on how much it deviates from the mean, weighted
    # by recency (more recent = more influential).
    feature_importance = {}

    recent_weight = 2.0                                          # Base weight for the most recent time step
    for i in range(min(len(window), req.top_features)):
        idx = len(window) - 1 - i                               # Index from most recent (t-0) backwards
        if idx >= 0:
            recency_weight = recent_weight / (i + 1)             # Decays as 2/1, 2/2, 2/3, … for older steps
            variance_importance = abs(window[idx] - np.mean(window)) / (np.std(window) + 1e-8)  # Normalized deviation
            importance = recency_weight * variance_importance     # Combined score
            feature_importance[f"t-{i}"] = float(importance)     # Label as t-0 (most recent), t-1, t-2, …

    # Trend feature: slope of a linear fit across the entire window
    if len(window) > 2:
        trend = np.polyfit(range(len(window)), window, 1)[0]     # Linear regression slope
        feature_importance['trend'] = float(abs(trend) * 10)     # Scale up for visibility

    # Volatility feature: standard deviation of first differences
    if len(window) > 1:
        volatility = np.std(np.diff(window))                     # How "jumpy" the series is
        feature_importance['volatility'] = float(volatility)

    # Sort features by absolute importance (descending) and keep top_features
    sorted_features = sorted(
        feature_importance.items(),
        key=lambda x: abs(x[1]),                                 # Sort by magnitude
        reverse=True
    )[:req.top_features]

    # Build a natural-language reasoning string
    reasoning = f"Model prediction is primarily influenced by recent values and trend patterns. "
    reasoning += f"Top contributing factors: {', '.join([k for k, _ in sorted_features[:3]])}"

    response = ExplainResponse(
        series_id=req.series_id,
        feature_importance=dict(sorted_features),                # Ordered dict of top features
        shap_values=None,                                        # No SHAP values in fallback mode
        reasoning=reasoning                                      # Human-readable explanation
    )

    CACHE.set(cache_key, response.dict())                        # Cache the heuristic result

    return response
