"""
Simple Standalone Training Script – LSTM Baseline for Chronos
===============================================================
This is a minimal, self-contained training script that trains a single LSTM
model on time-series data from a CSV file. It is intentionally verbose and
straightforward for learning and interview preparation.

Pipeline executed by this script:
  1. Load CSV via preprocess.load_timeseries().
  2. Pivot into per-series dict via preprocess.pivot_series().
  3. Resample each series to a regular frequency and fill gaps.
  4. Concatenate all series into one long 1-D array.
  5. Z-score normalize using global mean (mu) and std (sigma).
  6. Create sliding-window (X, y) pairs via SequenceDataset.
  7. Split 90/10 (train/val) — no separate test set in this simple version.
  8. Train the LSTM using a standard PyTorch training loop.
  9. Save the best checkpoint (lowest val loss) with mu and sigma for later
     denormalization during inference.

Usage:
    python -m chronos.training --data data/sample_timeseries.csv --epochs 10
"""

import argparse  # CLI argument parsing
import os        # File-system helpers (makedirs)
import numpy as np
import pandas as pd
import torch
from torch import nn, optim           # Loss functions and optimizers
from torch.utils.data import Dataset, DataLoader  # Data primitives

# Import basic preprocessing utilities from the standalone preprocess module.
from chronos.preprocess import load_timeseries, pivot_series, resample_and_fill, sliding_windows

# Import the LSTM model factory from the models package.
from chronos.models.lstm_model import make_model


class SequenceDataset(Dataset):
    """PyTorch Dataset that wraps sliding-window (X, y) pairs.

    Given a 1-D normalized time-series array, this dataset creates overlapping
    windows of length `input_len` as inputs and the next `horizon` values as
    targets. This is identical in purpose to training.trainer.SequenceDataset
    but kept here for standalone simplicity.
    """

    def __init__(self, series_values, input_len=60, horizon=1):
        """Build sliding-window samples from a 1-D array.

        Args:
            series_values: 1-D numpy array of z-score-normalized values.
            input_len: Number of past timesteps the model receives.
            horizon: Number of future timesteps to predict (1 for this baseline).
        """
        # sliding_windows returns X of shape (N, input_len) and y of shape (N, horizon).
        x, y = sliding_windows(series_values, input_len=input_len, horizon=horizon)

        # Cast to float32 for PyTorch compatibility.
        self.x = x.astype(np.float32)

        # Squeeze y from (N, 1) to (N,) since we only predict a single step.
        self.y = y.squeeze().astype(np.float32)

    def __len__(self):
        """Return the total number of samples."""
        return len(self.x)

    def __getitem__(self, idx):
        """Return one (input, target) pair.

        Input is reshaped to (input_len, 1) by adding a feature dimension
        with [:, None] — LSTM expects (seq_len, n_features).
        """
        return self.x[idx][:, None], self.y[idx]


def train_loop(dataloader, model, loss_fn, optimizer, device):
    """Run one training epoch over the entire DataLoader.

    Performs the standard PyTorch training step for each mini-batch:
    forward pass → compute loss → zero gradients → backprop → update weights.

    Args:
        dataloader: PyTorch DataLoader yielding (input, target) batches.
        model: The LSTM model.
        loss_fn: Loss function (MSELoss).
        optimizer: Optimizer (Adam).
        device: torch.device (CPU or CUDA).

    Returns:
        Average training loss per sample for this epoch.
    """
    # Set model to training mode (enables dropout, etc.).
    model.train()
    total_loss = 0.0  # Accumulate batch losses weighted by batch size.

    for xb, yb in dataloader:
        # Move data to the target device.
        xb = xb.to(device)
        yb = yb.to(device)

        # Forward pass through the model.
        pred = model(xb)

        # Compute MSE loss; squeeze pred from (batch, 1) to (batch,).
        loss = loss_fn(pred.squeeze(), yb)

        # Zero accumulated gradients from the previous step.
        optimizer.zero_grad()

        # Backpropagate to compute gradients of loss w.r.t. all parameters.
        loss.backward()

        # Update model weights.
        optimizer.step()

        # Accumulate weighted loss for correct per-sample averaging.
        total_loss += loss.item() * xb.size(0)

    # Return average loss across the entire dataset.
    return total_loss / len(dataloader.dataset)


def eval_loop(dataloader, model, loss_fn, device):
    """Evaluate the model on a validation DataLoader (no gradient computation).

    Args:
        dataloader: PyTorch DataLoader for the validation split.
        model: The LSTM model.
        loss_fn: Loss function (MSELoss).
        device: torch.device (CPU or CUDA).

    Returns:
        Average validation loss per sample.
    """
    # Set model to evaluation mode (disables dropout, freezes batch-norm).
    model.eval()
    total_loss = 0.0

    # torch.no_grad() disables gradient tracking, saving memory and computation.
    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)

            loss = loss_fn(pred.squeeze(), yb)

            total_loss += loss.item() * xb.size(0)

    return total_loss / len(dataloader.dataset)


def main(args):
    """Main training function — coordinates data loading, training, and saving.

    This function:
      1. Loads and preprocesses the time-series data.
      2. Normalizes it globally (z-score).
      3. Builds the Dataset and DataLoader.
      4. Trains the LSTM model.
      5. Saves the best checkpoint with normalization parameters.
    """
    # Ensure the output directory for the checkpoint exists.
    os.makedirs(os.path.dirname(args.model_out) or '.', exist_ok=True)

    # Step 1: Load the CSV into a DataFrame.
    df = load_timeseries(args.data)

    # Step 2: Pivot from long format to a dict of {series_id: pd.Series}.
    series = pivot_series(df)

    # Step 3: Resample every series to the requested frequency, fill NaN gaps,
    # and concatenate all values into one long 1-D array.
    all_vals = None
    for sid, s in series.items():
        # Resample to regular intervals and fill gaps via forward-fill + interpolation.
        s2 = resample_and_fill(s, freq=args.freq)
        vals = s2.values

        # Concatenate values from all series into a single array.
        if all_vals is None:
            all_vals = vals
        else:
            all_vals = np.concatenate([all_vals, vals])

    # Step 4: Compute global z-score normalization parameters.
    mu = all_vals.mean()     # Global mean of all time-series values.
    sigma = all_vals.std()   # Global standard deviation.
    print(f"global mean={mu:.4f} std={sigma:.4f}")

    # Step 5: Normalize the data and create the SequenceDataset.
    # (all_vals - mu) / sigma produces z-scores with mean≈0 and std≈1.
    dataset = SequenceDataset((all_vals - mu)/sigma, input_len=args.input_len, horizon=1)
    n = len(dataset)  # Total number of sliding-window samples.

    # Step 6: Chronological 90/10 split (no shuffle to avoid temporal leakage).
    split = int(n*0.9)
    train_ds = torch.utils.data.Subset(dataset, list(range(0, split)))
    val_ds = torch.utils.data.Subset(dataset, list(range(split, n)))

    # Step 7: Wrap in DataLoaders. Training shuffled for SGD; validation is not.
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # Step 8: Set up model, loss, and optimizer.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build the LSTM model and move it to the chosen device.
    model = make_model(input_size=1, hidden_size=args.hidden, num_layers=2, out_steps=1).to(device)

    # Mean Squared Error — standard loss for regression tasks.
    loss_fn = nn.MSELoss()

    # Adam optimizer with the specified learning rate.
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Step 9: Training loop — iterate over epochs and save the best model.
    best_val = float('inf')  # Track the lowest validation loss seen so far.

    for epoch in range(args.epochs):
        # Train for one epoch.
        train_loss = train_loop(train_loader, model, loss_fn, optimizer, device)

        # Evaluate on the validation set.
        val_loss = eval_loop(val_loader, model, loss_fn, device)

        print(f"Epoch {epoch+1}/{args.epochs} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        # Save checkpoint whenever validation loss improves.
        if val_loss < best_val:
            best_val = val_loss
            # The checkpoint includes model weights plus mu and sigma so that
            # inference.py can denormalize predictions back to the original scale.
            torch.save({'model_state': model.state_dict(), 'mu':mu, 'sigma':sigma}, args.model_out)

    print(f"Saved model to {args.model_out}")


if __name__ == '__main__':
    # ---- CLI argument definitions ----
    p = argparse.ArgumentParser()

    # Path to the input CSV file.
    p.add_argument('--data', default='data/sample_timeseries.csv')

    # Path where the best model checkpoint will be saved.
    p.add_argument('--model_out', default='artifacts/lstm_model.pth')

    # Number of past timesteps used as input (look-back window).
    p.add_argument('--input_len', type=int, default=60)

    # Resampling frequency (e.g. '1T' = 1 minute, '1H' = 1 hour).
    p.add_argument('--freq', default='1T')

    # Number of hidden units in each LSTM layer.
    p.add_argument('--hidden', type=int, default=64)

    # Number of training epochs.
    p.add_argument('--epochs', type=int, default=5)

    # Mini-batch size for DataLoader.
    p.add_argument('--batch_size', type=int, default=64)

    # Learning rate for the Adam optimizer.
    p.add_argument('--lr', type=float, default=1e-3)

    args = p.parse_args()
    main(args)
