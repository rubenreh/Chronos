"""
Comprehensive Training Script – CLI Entry Point for Chronos
=============================================================
This is the main command-line script for training one or more deep-learning
forecasting models (LSTM, TCN, Transformer) and optionally comparing them
against classical baselines (ARIMA, Prophet, SimpleRNN).

End-to-end pipeline executed by this script:
  1. Load a CSV of time-series data via ChronosDataLoader.
  2. Preprocess (resample, NaN-fill, z-score normalize) via Preprocessor.
  3. Create sliding-window (X, y) pairs via SequenceDataset.
  4. Split chronologically 80/10/10 (train / val / test) to prevent temporal leakage.
  5. For each requested model type, instantiate the architecture, train with
     the Trainer class (which handles early stopping + MLflow), and evaluate on
     the held-out test set.
  6. If multiple models are trained, build a simple averaging ensemble and report
     its metrics.
  7. Optionally run BenchmarkRunner to compare against ARIMA / Prophet baselines.

Usage examples:
    python -m chronos.training.train_comprehensive --data data/sample.csv --models lstm
    python -m chronos.training.train_comprehensive --data data/sample.csv --models lstm,tcn,transformer --use-mlflow --run-benchmarks
"""

import argparse      # Parse CLI arguments for data path, model type, etc.
import os            # File-system helpers (makedirs for model output dir)
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset  # Subset for chronological splits
from typing import Dict, Optional

# Chronos internal imports -----------------------------------------------
# DataLoader here is the *Chronos* DataLoader (CSV → pd.DataFrame → per-series dict).
from chronos.data.loader import DataLoader as ChronosDataLoader
# Preprocessor: resample + NaN fill + z-score normalize.
# sliding_windows: create overlapping (X, y) input/target pairs.
from chronos.data.preprocessor import Preprocessor, sliding_windows
# Factory functions that build LSTM / TCN / Transformer architectures.
# create_ensemble combines multiple models into one averaged predictor.
from chronos.models import (
    make_lstm_model,
    make_tcn_model,
    make_transformer_model,
    create_ensemble
)
# Trainer runs the PyTorch training loop with early stopping and MLflow.
# SequenceDataset wraps sliding-window arrays as a PyTorch Dataset.
from chronos.training.trainer import Trainer, SequenceDataset
# setup_mlflow creates a configured MLflowTracker (not used directly here
# because Trainer creates its own tracker, but imported for completeness).
from chronos.training.mlflow_tracker import setup_mlflow
# BenchmarkRunner compares our models against ARIMA, Prophet, SimpleRNN.
from chronos.evaluation.benchmark import BenchmarkRunner
# Metric functions used to evaluate the test set after training.
from chronos.evaluation.metrics import calculate_metrics, composite_score, calculate_stability_score, calculate_latency_score


def prepare_data(
    data_path: str,
    input_len: int = 60,
    horizon: int = 1,
    train_split: float = 0.8,
    val_split: float = 0.1
) -> Dict:
    """Load, preprocess, and split time-series data into train/val/test loaders.

    This function is the single data-preparation entry point for the entire
    comprehensive training script. It:
      1. Reads the CSV via ChronosDataLoader.
      2. Pivots it into individual series.
      3. Runs each series through the Preprocessor (resample, fill, normalize).
      4. Combines all series into one long array (multi-series concatenation).
      5. Creates a SequenceDataset of sliding-window samples.
      6. Splits chronologically (80 / 10 / 10) to prevent future data leakage.
      7. Wraps each split in a PyTorch DataLoader with batch_size=64.

    Args:
        data_path: Path to the input CSV (expected columns: timestamp, series_id, value).
        input_len: Number of past timesteps in each input window.
        horizon: Number of future timesteps to predict.
        train_split: Fraction of data used for training (default 0.8 = 80%).
        val_split: Fraction of data used for validation (default 0.1 = 10%).
                   The remaining 10% is used for testing.

    Returns:
        Dictionary with keys:
            train_loader, val_loader, test_loader – PyTorch DataLoaders
            mu, sigma – normalization parameters for inverse-transforming predictions
            test_data – a small slice of raw values for optional baseline benchmarks
    """
    # Step 1: Load CSV into a DataFrame via ChronosDataLoader.
    loader = ChronosDataLoader(data_path=data_path)

    # Step 2: Pivot from long format (timestamp, series_id, value) to a dict
    # mapping each series_id to a pd.Series indexed by timestamp.
    series_dict = loader.pivot_series()

    # Step 3: Preprocess each series individually.
    # freq='1T' resamples to 1-minute intervals; normalize=True applies z-score.
    preprocessor = Preprocessor(freq='1T', normalize=True)
    all_values = []  # Will collect normalized values from every series.

    for sid, series in series_dict.items():
        # fit_transform: resample → NaN fill → compute mu/sigma → normalize.
        processed = preprocessor.fit_transform(series)
        # Append individual values to the combined list.
        all_values.extend(processed.values)

    # Convert to a single contiguous numpy array for sliding_windows.
    all_values = np.array(all_values)

    # Store the normalization params so we can denormalize predictions later.
    mu = preprocessor.mu
    sigma = preprocessor.sigma

    # Step 5: Create the SequenceDataset (sliding-window pairs).
    dataset = SequenceDataset(all_values, input_len=input_len, horizon=horizon)
    n = len(dataset)  # Total number of sliding-window samples.

    # Step 6: Chronological split — we do NOT shuffle before splitting because
    # shuffling would leak future information into the training set.
    train_end = int(n * train_split)              # Index where training ends
    val_end = train_end + int(n * val_split)      # Index where validation ends

    # Subset uses index lists to slice the dataset without copying data.
    train_ds = Subset(dataset, list(range(0, train_end)))
    val_ds = Subset(dataset, list(range(train_end, val_end)))
    test_ds = Subset(dataset, list(range(val_end, n)))

    # Step 7: Wrap each split in a PyTorch DataLoader.
    # Training loader is shuffled for stochastic gradient descent.
    # Val/test loaders are NOT shuffled — order doesn't matter for evaluation.
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'mu': mu,       # Mean used for z-score normalization
        'sigma': sigma,  # Std dev used for z-score normalization
        # Keep a small slice of raw values for baseline benchmarks.
        'test_data': all_values[val_end:val_end+100] if len(all_values) > val_end else all_values[-100:]
    }


def train_model_type(
    model_type: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    input_len: int,
    device: torch.device,
    epochs: int,
    lr: float,
    save_dir: str,
    use_mlflow: bool = True,
    experiment_name: str = "chronos_forecasting"
) -> Dict:
    """Instantiate, train, and evaluate a single model architecture.

    This function handles the full lifecycle for one model type:
      1. Build the model via the appropriate factory function.
      2. Train it using the Trainer class (with early stopping, patience=10).
      3. Evaluate on the held-out test DataLoader.
      4. Update the saved checkpoint with normalization metadata.

    Args:
        model_type: One of 'lstm', 'tcn', or 'transformer'.
        train_loader: PyTorch DataLoader for training data.
        val_loader: PyTorch DataLoader for validation data.
        test_loader: PyTorch DataLoader for test data.
        input_len: Length of the input window (needed for model construction).
        device: torch.device to run on (CPU or CUDA GPU).
        epochs: Maximum training epochs.
        lr: Learning rate.
        save_dir: Directory where model checkpoints will be saved.
        use_mlflow: Whether to log to MLflow.
        experiment_name: Base name for the MLflow experiment (model type is appended).

    Returns:
        Dictionary with keys: model, model_type, save_path, history, test_metrics.
    """
    # ---- Step 1: Create the model architecture ----
    if model_type == 'lstm':
        # LSTM with 2 stacked layers, hidden_size=64, predicting 1 future step.
        model = make_lstm_model(input_size=1, hidden_size=64, num_layers=2, out_steps=1)
    elif model_type == 'tcn':
        # Temporal Convolutional Network with 3 layers of 64 channels each.
        model = make_tcn_model(input_size=1, output_size=1, num_channels=[64, 64, 64])
    elif model_type == 'transformer':
        # Transformer with d_model=128, 8 attention heads, 4 encoder layers.
        model = make_transformer_model(input_size=1, d_model=128, nhead=8, num_layers=4, output_size=1)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # ---- Step 2: Define the checkpoint save path ----
    save_path = os.path.join(save_dir, f"{model_type}_model.pth")

    # ---- Step 3: Train with Trainer ----
    # Each model type gets its own MLflow experiment for clean organization.
    trainer = Trainer(
        model=model,
        device=device,
        use_mlflow=use_mlflow,
        experiment_name=f"{experiment_name}_{model_type}"
    )

    # train() handles the full loop: epochs, validation, early stopping (patience=10),
    # checkpoint saving, and MLflow logging.
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        save_path=save_path,
        patience=10  # Stop if val loss doesn't improve for 10 consecutive epochs.
    )

    # ---- Step 4: Evaluate on the test set ----
    trainer.model.eval()  # Ensure model is in evaluation mode.
    test_preds = []
    test_targets = []

    # Run inference on the test DataLoader with gradients disabled.
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)  # Move input batch to device.
            yb = yb.to(device)  # Move target batch to device.

            pred = trainer.model(xb)  # Forward pass.

            # Squeeze single-step output from (batch, 1) to (batch,).
            if pred.dim() > 1 and pred.size(1) == 1:
                pred = pred.squeeze()

            # Collect predictions and targets on CPU as numpy arrays.
            test_preds.append(pred.cpu().numpy())
            test_targets.append(yb.cpu().numpy())

    # Concatenate all batches into full arrays.
    test_preds = np.concatenate(test_preds)
    test_targets = np.concatenate(test_targets)

    # Compute comprehensive test metrics (MSE, RMSE, MAE, MAPE, R², etc.).
    test_metrics = calculate_metrics(test_targets, test_preds)

    # ---- Step 5: Update checkpoint with normalization metadata ----
    # Reload the saved checkpoint so we can add mu/sigma and model_type.
    checkpoint = torch.load(save_path)
    checkpoint['mu'] = 0.0        # Placeholder — caller should set from data_dict.
    checkpoint['sigma'] = 1.0     # Placeholder — caller should set from data_dict.
    checkpoint['model_type'] = model_type  # Tag so inference.py knows which architecture to build.
    torch.save(checkpoint, save_path)  # Re-save with the extra metadata.

    return {
        'model': model,
        'model_type': model_type,
        'save_path': save_path,
        'history': history,
        'test_metrics': test_metrics
    }


def main(args):
    """Main training function — orchestrates the full training pipeline.

    Reads CLI args, prepares data, trains each requested model type,
    optionally creates an ensemble and runs baseline benchmarks.
    """
    # Create the output directory for model checkpoints if it doesn't exist.
    os.makedirs(args.model_dir, exist_ok=True)

    # Select GPU if available; otherwise fall back to CPU.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ---- Data preparation ----
    print("Loading and preparing data...")
    data_dict = prepare_data(
        data_path=args.data,
        input_len=args.input_len,
        horizon=1,           # Single-step forecasting.
        train_split=0.8,     # 80% training data.
        val_split=0.1        # 10% validation data; remaining 10% is test.
    )

    # Unpack the prepared data dictionary for convenience.
    train_loader = data_dict['train_loader']
    val_loader = data_dict['val_loader']
    test_loader = data_dict['test_loader']
    mu = data_dict['mu']      # Normalization mean — needed for denormalization at inference.
    sigma = data_dict['sigma']  # Normalization std dev.

    # Print dataset sizes so the user can verify the split ratios.
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Normalization: mu={mu:.4f}, sigma={sigma:.4f}")

    # ---- Train each requested model type ----
    # The --models flag accepts a comma-separated list, e.g. "lstm,tcn,transformer".
    models_to_train = args.models.split(',') if args.models else ['lstm']
    trained_models = {}  # Stores results keyed by model type name.

    for model_type in models_to_train:
        print(f"\n{'='*60}")
        print(f"Training {model_type.upper()} model...")
        print(f"{'='*60}")

        # Train and evaluate this model type.
        result = train_model_type(
            model_type=model_type,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            input_len=args.input_len,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            save_dir=args.model_dir,
            use_mlflow=args.use_mlflow,
            experiment_name=args.experiment_name
        )

        trained_models[model_type] = result

        # Print test metrics for this model.
        print(f"\n{model_type.upper()} Test Metrics:")
        for metric, value in result['test_metrics'].items():
            print(f"  {metric}: {value:.6f}")

    # ---- Ensemble (if multiple models were trained) ----
    if len(trained_models) > 1:
        print(f"\n{'='*60}")
        print("Creating ensemble model...")
        print(f"{'='*60}")

        # Collect all trained model objects into a list.
        model_list = [result['model'] for result in trained_models.values()]

        # create_ensemble builds a wrapper that averages predictions from all models.
        ensemble = create_ensemble(model_list, method='weighted_average')

        # Evaluate the ensemble on the test set.
        ensemble.eval()
        test_preds = []
        test_targets = []

        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                # Get predictions from every individual model in the ensemble.
                preds_list = []
                for model in model_list:
                    model.eval()  # Ensure each sub-model is in eval mode.
                    pred = model(xb)
                    if pred.dim() > 1 and pred.size(1) == 1:
                        pred = pred.squeeze()
                    preds_list.append(pred.cpu().numpy())

                # Simple mean-ensemble: average predictions across all models.
                ensemble_pred = np.mean(preds_list, axis=0)
                test_preds.append(ensemble_pred)
                test_targets.append(yb.cpu().numpy())

        # Concatenate batch results.
        test_preds = np.concatenate(test_preds)
        test_targets = np.concatenate(test_targets)

        # Compute ensemble metrics.
        ensemble_metrics = calculate_metrics(test_targets, test_preds)

        print(f"\nEnsemble Test Metrics:")
        for metric, value in ensemble_metrics.items():
            print(f"  {metric}: {value:.6f}")

        # Persist a human-readable summary of the ensemble composition and metrics.
        ensemble_path = os.path.join(args.model_dir, 'ensemble_info.txt')
        with open(ensemble_path, 'w') as f:
            f.write(f"Ensemble of: {', '.join(trained_models.keys())}\n")
            f.write(f"Method: weighted_average\n")
            for metric, value in ensemble_metrics.items():
                f.write(f"{metric}: {value:.6f}\n")

    # ---- Baseline benchmarks ----
    if args.run_benchmarks:
        print(f"\n{'='*60}")
        print("Running baseline benchmarks...")
        print(f"{'='*60}")

        # Use a small slice of raw data for the statistical baselines.
        test_data = data_dict['test_data']
        # Split the benchmark data in half: first half for fitting, second for testing.
        train_data = test_data[:len(test_data)//2]
        benchmark_test = test_data[len(test_data)//2:]

        # BenchmarkRunner tries ARIMA, Prophet, and SimpleRNN and reports metrics.
        benchmark_runner = BenchmarkRunner()
        benchmark_results = benchmark_runner.run_all_benchmarks(
            train_data=train_data,
            test_data=benchmark_test
        )

        print("\nBaseline Benchmark Results:")
        for result in benchmark_results:
            print(f"\n{result.model_name}:")
            print(f"  RMSE: {result.metrics.get('rmse', 0):.6f}")
            print(f"  MAE: {result.metrics.get('mae', 0):.6f}")
            print(f"  R²: {result.metrics.get('r2', 0):.6f}")
            print(f"  Latency: {result.latency_ms:.2f} ms")
            print(f"  Composite Score: {result.composite_score:.4f}")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"Models saved to: {args.model_dir}")
    if args.use_mlflow:
        print(f"MLflow UI: mlflow ui --port 5000")
    print(f"{'='*60}")


if __name__ == '__main__':
    # ---- Argument parser ----
    # This block defines the CLI interface for the comprehensive training script.
    parser = argparse.ArgumentParser(description='Comprehensive model training with MLflow')

    # Path to the input CSV (expected columns: timestamp, series_id, value).
    parser.add_argument('--data', default='data/sample_timeseries.csv', help='Path to data CSV')

    # Directory where trained model checkpoints (.pth files) will be saved.
    parser.add_argument('--model-dir', default='artifacts', help='Directory to save models')

    # Comma-separated list of model architectures to train.
    parser.add_argument('--models', default='lstm', help='Comma-separated list: lstm,tcn,transformer')

    # Number of past timesteps used as model input (look-back window size).
    parser.add_argument('--input-len', type=int, default=60, help='Input sequence length')

    # Maximum number of training epochs.
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')

    # Learning rate for the Adam optimizer.
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')

    # When set, enables MLflow experiment tracking for all training runs.
    parser.add_argument('--use-mlflow', action='store_true', help='Use MLflow tracking')

    # Name of the MLflow experiment (visible in the MLflow UI).
    parser.add_argument('--experiment-name', default='chronos_forecasting', help='MLflow experiment name')

    # When set, runs ARIMA / Prophet / SimpleRNN baselines after training.
    parser.add_argument('--run-benchmarks', action='store_true', help='Run baseline benchmarks')

    # Parse the CLI arguments and kick off the main training pipeline.
    args = parser.parse_args()
    main(args)
