"""Comprehensive training script with MLflow integration, baseline benchmarks, and model comparison."""
import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from typing import Dict, Optional

from chronos.data.loader import DataLoader as ChronosDataLoader
from chronos.data.preprocessor import Preprocessor, sliding_windows
from chronos.models import (
    make_lstm_model,
    make_tcn_model,
    make_transformer_model,
    create_ensemble
)
from chronos.training.trainer import Trainer, SequenceDataset
from chronos.training.mlflow_tracker import setup_mlflow
from chronos.evaluation.benchmark import BenchmarkRunner
from chronos.evaluation.metrics import calculate_metrics, composite_score, calculate_stability_score, calculate_latency_score


def prepare_data(
    data_path: str,
    input_len: int = 60,
    horizon: int = 1,
    train_split: float = 0.8,
    val_split: float = 0.1
) -> Dict:
    """Prepare data for training.
    
    Returns:
        Dictionary with train/val/test loaders and normalization params
    """
    # Load data
    loader = ChronosDataLoader(data_path=data_path)
    series_dict = loader.pivot_series()
    
    # Combine all series for training
    preprocessor = Preprocessor(freq='1T', normalize=True)
    all_values = []
    
    for sid, series in series_dict.items():
        processed = preprocessor.fit_transform(series)
        all_values.extend(processed.values)
    
    all_values = np.array(all_values)
    
    # Get normalization params from preprocessor
    mu = preprocessor.mu
    sigma = preprocessor.sigma
    
    # Create dataset
    dataset = SequenceDataset(all_values, input_len=input_len, horizon=horizon)
    n = len(dataset)
    
    # Split
    train_end = int(n * train_split)
    val_end = train_end + int(n * val_split)
    
    train_ds = Subset(dataset, list(range(0, train_end)))
    val_ds = Subset(dataset, list(range(train_end, val_end)))
    test_ds = Subset(dataset, list(range(val_end, n)))
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'mu': mu,
        'sigma': sigma,
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
    """Train a specific model type.
    
    Returns:
        Dictionary with model, metrics, and save path
    """
    # Create model
    if model_type == 'lstm':
        model = make_lstm_model(input_size=1, hidden_size=64, num_layers=2, out_steps=1)
    elif model_type == 'tcn':
        model = make_tcn_model(input_size=1, output_size=1, num_channels=[64, 64, 64])
    elif model_type == 'transformer':
        model = make_transformer_model(input_size=1, d_model=128, nhead=8, num_layers=4, output_size=1)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Save path
    save_path = os.path.join(save_dir, f"{model_type}_model.pth")
    
    # Train
    trainer = Trainer(
        model=model,
        device=device,
        use_mlflow=use_mlflow,
        experiment_name=f"{experiment_name}_{model_type}"
    )
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        save_path=save_path,
        patience=10
    )
    
    # Evaluate on test set
    trainer.model.eval()
    test_preds = []
    test_targets = []
    
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = trainer.model(xb)
            if pred.dim() > 1 and pred.size(1) == 1:
                pred = pred.squeeze()
            test_preds.append(pred.cpu().numpy())
            test_targets.append(yb.cpu().numpy())
    
    test_preds = np.concatenate(test_preds)
    test_targets = np.concatenate(test_targets)
    
    test_metrics = calculate_metrics(test_targets, test_preds)
    
    # Save model with normalization params
    checkpoint = torch.load(save_path)
    checkpoint['mu'] = 0.0  # Will be set from data prep
    checkpoint['sigma'] = 1.0
    checkpoint['model_type'] = model_type
    torch.save(checkpoint, save_path)
    
    return {
        'model': model,
        'model_type': model_type,
        'save_path': save_path,
        'history': history,
        'test_metrics': test_metrics
    }


def main(args):
    """Main training function."""
    os.makedirs(args.model_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Prepare data
    print("Loading and preparing data...")
    data_dict = prepare_data(
        data_path=args.data,
        input_len=args.input_len,
        horizon=1,
        train_split=0.8,
        val_split=0.1
    )
    
    train_loader = data_dict['train_loader']
    val_loader = data_dict['val_loader']
    test_loader = data_dict['test_loader']
    mu = data_dict['mu']
    sigma = data_dict['sigma']
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Normalization: mu={mu:.4f}, sigma={sigma:.4f}")
    
    # Train models
    models_to_train = args.models.split(',') if args.models else ['lstm']
    trained_models = {}
    
    for model_type in models_to_train:
        print(f"\n{'='*60}")
        print(f"Training {model_type.upper()} model...")
        print(f"{'='*60}")
        
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
        
        print(f"\n{model_type.upper()} Test Metrics:")
        for metric, value in result['test_metrics'].items():
            print(f"  {metric}: {value:.6f}")
    
    # Create ensemble if multiple models
    if len(trained_models) > 1:
        print(f"\n{'='*60}")
        print("Creating ensemble model...")
        print(f"{'='*60}")
        
        model_list = [result['model'] for result in trained_models.values()]
        ensemble = create_ensemble(model_list, method='weighted_average')
        
        # Evaluate ensemble
        ensemble.eval()
        test_preds = []
        test_targets = []
        
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                
                # Get predictions from all models
                preds_list = []
                for model in model_list:
                    model.eval()
                    pred = model(xb)
                    if pred.dim() > 1 and pred.size(1) == 1:
                        pred = pred.squeeze()
                    preds_list.append(pred.cpu().numpy())
                
                # Ensemble prediction
                ensemble_pred = np.mean(preds_list, axis=0)
                test_preds.append(ensemble_pred)
                test_targets.append(yb.cpu().numpy())
        
        test_preds = np.concatenate(test_preds)
        test_targets = np.concatenate(test_targets)
        
        ensemble_metrics = calculate_metrics(test_targets, test_preds)
        
        print(f"\nEnsemble Test Metrics:")
        for metric, value in ensemble_metrics.items():
            print(f"  {metric}: {value:.6f}")
        
        # Save ensemble info
        ensemble_path = os.path.join(args.model_dir, 'ensemble_info.txt')
        with open(ensemble_path, 'w') as f:
            f.write(f"Ensemble of: {', '.join(trained_models.keys())}\n")
            f.write(f"Method: weighted_average\n")
            for metric, value in ensemble_metrics.items():
                f.write(f"{metric}: {value:.6f}\n")
    
    # Run baseline benchmarks if requested
    if args.run_benchmarks:
        print(f"\n{'='*60}")
        print("Running baseline benchmarks...")
        print(f"{'='*60}")
        
        # Get test data for benchmarks
        test_data = data_dict['test_data']
        train_data = test_data[:len(test_data)//2]
        benchmark_test = test_data[len(test_data)//2:]
        
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
    
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"Models saved to: {args.model_dir}")
    if args.use_mlflow:
        print(f"MLflow UI: mlflow ui --port 5000")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Comprehensive model training with MLflow')
    parser.add_argument('--data', default='data/sample_timeseries.csv', help='Path to data CSV')
    parser.add_argument('--model-dir', default='artifacts', help='Directory to save models')
    parser.add_argument('--models', default='lstm', help='Comma-separated list: lstm,tcn,transformer')
    parser.add_argument('--input-len', type=int, default=60, help='Input sequence length')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--use-mlflow', action='store_true', help='Use MLflow tracking')
    parser.add_argument('--experiment-name', default='chronos_forecasting', help='MLflow experiment name')
    parser.add_argument('--run-benchmarks', action='store_true', help='Run baseline benchmarks')
    
    args = parser.parse_args()
    main(args)

