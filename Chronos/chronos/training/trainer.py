"""
Trainer Module – Core Training Loop for Chronos
=================================================
This module contains:

  • SequenceDataset – a PyTorch Dataset that converts a 1-D time-series array
                      into (X, y) pairs via the sliding-window technique.
  • Trainer         – the main training orchestrator. It runs the epoch loop,
                      evaluates on the validation set, implements early stopping,
                      saves the best model checkpoint, and (optionally) logs
                      every metric to MLflow through MLflowTracker.
  • train_model     – a convenience function that wraps Trainer for callers
                      who just need a single function call to kick off training.

Data flow inside the training pipeline:
  CSV → DataLoader (loader.py) → Preprocessor (resample, NaN-fill, z-score) →
  sliding_windows() → SequenceDataset → PyTorch DataLoader → Model → Trainer
"""

import os       # File-system operations (makedirs for checkpoint dirs)
import time     # Wall-clock timing for per-epoch duration
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader  # PyTorch data primitives
from typing import Dict, Optional, Any, List

# Chronos internal imports -----------------------------------------------
# ChronosDataLoader loads CSVs and pivots them into per-series pd.Series.
from chronos.data.loader import DataLoader as ChronosDataLoader
# Preprocessor handles resampling, NaN filling, and z-score normalization.
# sliding_windows creates overlapping (input, target) pairs from a 1-D array.
from chronos.data.preprocessor import Preprocessor, sliding_windows
# MLflowTracker wraps MLflow calls (start_run, log_params, log_metrics, etc.).
from chronos.training.mlflow_tracker import MLflowTracker
# calculate_metrics computes MSE, RMSE, MAE, MAPE, directional accuracy, R².
from chronos.evaluation.metrics import calculate_metrics


class SequenceDataset(Dataset):
    """PyTorch Dataset that converts a 1-D time-series array into
    (input_window, target) pairs using a sliding-window approach.

    Each sample is:
        X  – a tensor of shape (input_len, 1) representing the look-back window
        y  – a scalar (horizon=1) or vector (horizon>1) of target values
    """

    def __init__(
        self,
        series_values: np.ndarray,
        input_len: int = 60,
        horizon: int = 1
    ):
        """Initialize dataset by slicing the time-series into overlapping windows.

        Args:
            series_values: 1-D numpy array of (already-normalized) time-series values.
            input_len: Number of past timesteps the model sees as input (look-back).
            horizon: Number of future timesteps the model must predict.
        """
        # sliding_windows returns (X, y) numpy arrays.
        # X shape: (num_windows, input_len)
        # y shape: (num_windows, horizon)
        x, y = sliding_windows(series_values, input_len=input_len, horizon=horizon)

        # Cast to float32 — PyTorch's default floating-point precision.
        self.x = x.astype(np.float32)

        # If we're predicting a single future step, squeeze y from (N, 1) to (N,)
        # so that the loss function receives matching tensor shapes.
        self.y = y.squeeze().astype(np.float32) if horizon == 1 else y.astype(np.float32)

    def __len__(self):
        """Return the total number of sliding-window samples."""
        return len(self.x)

    def __getitem__(self, idx):
        """Return one (input, target) pair.

        The input is reshaped from (input_len,) → (input_len, 1) by adding a
        feature dimension with [:, None], because PyTorch sequential models
        (LSTM, TCN, Transformer) expect shape (seq_len, n_features).
        """
        return self.x[idx][:, None], self.y[idx]


class Trainer:
    """Model trainer with MLflow integration and early stopping.

    Responsibilities:
      1. Run the training loop for N epochs.
      2. Evaluate on a validation DataLoader after every epoch.
      3. Optionally log metrics/params to MLflow via MLflowTracker.
      4. Save the best model checkpoint (lowest val loss) to disk.
      5. Implement patience-based early stopping to prevent overfitting.
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        use_mlflow: bool = True,
        experiment_name: str = "chronos_forecasting"
    ):
        """Initialize trainer.

        Args:
            model: PyTorch model to train (LSTM, TCN, or Transformer).
            device: Device to train on. Defaults to CUDA if a GPU is available,
                    otherwise falls back to CPU.
            use_mlflow: When True, all metrics and artifacts are logged to MLflow.
            experiment_name: The MLflow experiment name used to group related runs.
        """
        self.model = model

        # Auto-detect GPU; fall back to CPU when no CUDA device is present.
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Move model weights to the chosen device (GPU or CPU).
        self.model.to(self.device)

        self.use_mlflow = use_mlflow

        # Create MLflowTracker only when tracking is requested.
        self.mlflow_tracker = MLflowTracker(experiment_name=experiment_name) if use_mlflow else None

        # Stores per-epoch train/val loss and metrics for post-training analysis.
        self.training_history = []

    def train_epoch(
        self,
        dataloader: DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer
    ) -> float:
        """Execute one full pass (epoch) over the training DataLoader.

        For each mini-batch this method performs the standard PyTorch training
        step: forward pass → loss computation → backpropagation → weight update.

        Returns:
            Average training loss across all samples in this epoch.
        """
        # Set model to training mode (enables dropout, batch-norm updates, etc.).
        self.model.train()

        total_loss = 0.0  # Accumulator for total weighted loss across batches
        n_samples = 0     # Accumulator for total number of samples processed

        for xb, yb in dataloader:
            # Move input and target tensors to the training device (GPU/CPU).
            xb = xb.to(self.device)
            yb = yb.to(self.device)

            # Forward pass: run the model on the input window.
            pred = self.model(xb)

            # If the model outputs shape (batch, 1), squeeze to (batch,) so it
            # matches yb's shape and the loss function works correctly.
            if pred.dim() > 1 and pred.size(1) == 1:
                pred = pred.squeeze()

            # Compute the loss between predictions and ground truth.
            loss = loss_fn(pred, yb)

            # Zero out the gradients from the previous step to avoid accumulation.
            optimizer.zero_grad()

            # Backpropagate: compute gradients of the loss w.r.t. model parameters.
            loss.backward()

            # Update model weights using the computed gradients.
            optimizer.step()

            # Accumulate loss weighted by batch size for correct averaging later.
            total_loss += loss.item() * xb.size(0)
            n_samples += xb.size(0)

        # Return the mean loss per sample. Guard against empty dataloader.
        return total_loss / n_samples if n_samples > 0 else 0.0

    def evaluate(
        self,
        dataloader: DataLoader,
        loss_fn: nn.Module
    ) -> Dict[str, float]:
        """Evaluate the model on a validation or test DataLoader.

        Runs inference with gradients disabled (torch.no_grad) for speed and
        lower memory usage, then computes comprehensive metrics via
        calculate_metrics (MSE, RMSE, MAE, MAPE, directional accuracy, R²).

        Returns:
            Dictionary of metric names → float values, including 'loss'.
        """
        # Switch to evaluation mode (disables dropout, freezes batch-norm stats).
        self.model.eval()

        total_loss = 0.0
        n_samples = 0
        all_preds = []    # Collect predictions across all batches for metric calc
        all_targets = []  # Collect ground-truth values across all batches

        # Disable gradient computation — we don't need it during evaluation,
        # and skipping it saves memory and speeds up the forward pass.
        with torch.no_grad():
            for xb, yb in dataloader:
                # Move batch tensors to the device.
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                # Forward pass only (no backprop).
                pred = self.model(xb)

                # Squeeze single-step outputs to match target shape.
                if pred.dim() > 1 and pred.size(1) == 1:
                    pred = pred.squeeze()

                # Compute loss for this batch.
                loss = loss_fn(pred, yb)

                # Accumulate weighted loss.
                total_loss += loss.item() * xb.size(0)
                n_samples += xb.size(0)

                # Move predictions and targets back to CPU and convert to numpy
                # for scikit-learn-based metric calculations.
                all_preds.append(pred.cpu().numpy())
                all_targets.append(yb.cpu().numpy())

        # Mean loss across the entire dataset.
        avg_loss = total_loss / n_samples if n_samples > 0 else 0.0

        # Concatenate batch-level arrays into full dataset-level arrays.
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        # calculate_metrics returns MSE, RMSE, MAE, MAPE, directional accuracy, R².
        metrics = calculate_metrics(all_targets, all_preds)

        # Include the raw loss so callers can use it for early stopping.
        metrics['loss'] = avg_loss

        return metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        lr: float = 1e-3,
        loss_fn: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        save_path: Optional[str] = None,
        patience: Optional[int] = None
    ) -> Dict[str, List[float]]:
        """Full training loop with validation, early stopping, and MLflow logging.

        This is the main entry point for training a Chronos model. It:
          1. Optionally starts an MLflow run and logs hyperparameters.
          2. Iterates over epochs, calling train_epoch + evaluate each time.
          3. Saves the best checkpoint when validation loss improves.
          4. Stops early when validation loss hasn't improved for `patience` epochs.
          5. Logs the final checkpoint artifact to MLflow.

        Args:
            train_loader: PyTorch DataLoader for the training split.
            val_loader: PyTorch DataLoader for the validation split.
            epochs: Maximum number of epochs to train.
            lr: Learning rate for the optimizer (used only if optimizer is None).
            loss_fn: Loss function. Defaults to MSELoss (standard for regression).
            optimizer: Optimizer. Defaults to Adam with the given lr.
            save_path: File path to save the best model checkpoint (.pth).
            patience: Number of epochs with no val-loss improvement before
                      stopping. Set to None to disable early stopping.

        Returns:
            History dict with keys 'train_loss', 'val_loss', 'val_metrics'.
        """
        # Default to Mean Squared Error — the standard regression loss.
        if loss_fn is None:
            loss_fn = nn.MSELoss()

        # Default to Adam — widely used adaptive-learning-rate optimizer.
        if optimizer is None:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Track the best validation loss seen so far for checkpointing.
        best_val_loss = float('inf')

        # Counter for how many consecutive epochs val loss has not improved.
        patience_counter = 0

        # Dictionaries that will hold the per-epoch history for later analysis.
        history = {'train_loss': [], 'val_loss': [], 'val_metrics': []}

        # ------- MLflow run management -------
        run = None
        if self.use_mlflow and self.mlflow_tracker:
            # Start a new MLflow run; everything logged until end_run() is
            # grouped under this run in the MLflow UI.
            run = self.mlflow_tracker.start_run()

            # Log static hyperparameters once at the beginning of the run.
            self.mlflow_tracker.log_params({
                'epochs': epochs,
                'lr': lr,
                'model_type': type(self.model).__name__,
                'device': str(self.device)
            })

        try:
            for epoch in range(epochs):
                # Record wall-clock time so we can log epoch duration.
                start_time = time.time()

                # ---- Training step ----
                train_loss = self.train_epoch(train_loader, loss_fn, optimizer)

                # ---- Validation step ----
                val_metrics = self.evaluate(val_loader, loss_fn)
                val_loss = val_metrics['loss']

                # How long this epoch took (seconds).
                epoch_time = time.time() - start_time

                # ---- Record history ----
                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                history['val_metrics'].append(val_metrics)

                # ---- Log to MLflow ----
                if self.use_mlflow and self.mlflow_tracker:
                    # Log per-epoch metrics with the epoch number as the step.
                    self.mlflow_tracker.log_metrics({
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        # Prefix validation metrics with 'val_' for clarity.
                        **{f'val_{k}': v for k, v in val_metrics.items() if k != 'loss'},
                        'epoch_time': epoch_time
                    }, step=epoch)

                # ---- Console output ----
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_loss:.6f}, "
                      f"Val Loss: {val_loss:.6f}, "
                      f"Val RMSE: {val_metrics.get('rmse', 0):.6f}")

                # ---- Checkpoint best model ----
                if val_loss < best_val_loss:
                    # Validation improved — update best and reset patience.
                    best_val_loss = val_loss
                    patience_counter = 0

                    if save_path:
                        # Ensure the directory exists before writing the file.
                        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

                        # Save a checkpoint dict containing the model weights,
                        # the epoch number, the best val loss, and the full
                        # validation metrics dictionary.
                        torch.save({
                            'model_state': self.model.state_dict(),
                            'epoch': epoch,
                            'val_loss': val_loss,
                            'val_metrics': val_metrics
                        }, save_path)
                else:
                    # Validation did NOT improve — increment patience counter.
                    patience_counter += 1

                    # If patience has been exceeded, stop training to avoid
                    # overfitting on the training set.
                    if patience and patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break

            # After training, log the best checkpoint as an MLflow artifact
            # so it can be retrieved from the MLflow UI / API later.
            if self.use_mlflow and self.mlflow_tracker and save_path:
                self.mlflow_tracker.log_artifact(save_path)

        finally:
            # Always close the MLflow run — even if an exception occurs — to
            # avoid leaving an orphaned "running" entry in the MLflow backend.
            if run:
                self.mlflow_tracker.end_run()

        # Store history on the Trainer instance so callers can access it later.
        self.training_history = history
        return history


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-3,
    save_path: Optional[str] = None,
    use_mlflow: bool = True
) -> Dict[str, List[float]]:
    """Convenience function to train a model in a single call.

    This is a thin wrapper around Trainer that creates a Trainer instance,
    calls train(), and returns the history. Useful for scripts and notebooks
    where you don't need to keep a reference to the Trainer object.

    Args:
        model: PyTorch model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        epochs: Maximum number of training epochs.
        lr: Learning rate.
        save_path: Where to save the best model checkpoint.
        use_mlflow: Whether to enable MLflow logging.

    Returns:
        History dictionary with train_loss, val_loss, val_metrics lists.
    """
    # Create a Trainer with defaults and delegate to its train() method.
    trainer = Trainer(model, use_mlflow=use_mlflow)
    return trainer.train(train_loader, val_loader, epochs=epochs, lr=lr, save_path=save_path)
