"""Training utilities with MLflow integration."""
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Optional, Any, List

from chronos.data.loader import DataLoader as ChronosDataLoader
from chronos.data.preprocessor import Preprocessor, sliding_windows
from chronos.training.mlflow_tracker import MLflowTracker
from chronos.evaluation.metrics import calculate_metrics


class SequenceDataset(Dataset):
    """Dataset for time-series sequences."""
    
    def __init__(
        self,
        series_values: np.ndarray,
        input_len: int = 60,
        horizon: int = 1
    ):
        """Initialize dataset.
        
        Args:
            series_values: 1D numpy array of time-series values
            input_len: Length of input sequence
            horizon: Number of steps to predict
        """
        x, y = sliding_windows(series_values, input_len=input_len, horizon=horizon)
        self.x = x.astype(np.float32)
        self.y = y.squeeze().astype(np.float32) if horizon == 1 else y.astype(np.float32)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx][:, None], self.y[idx]


class Trainer:
    """Model trainer with MLflow integration."""
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        use_mlflow: bool = True,
        experiment_name: str = "chronos_forecasting"
    ):
        """Initialize trainer.
        
        Args:
            model: PyTorch model to train
            device: Device to train on (defaults to cuda if available)
            use_mlflow: Whether to use MLflow tracking
            experiment_name: MLflow experiment name
        """
        self.model = model
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.use_mlflow = use_mlflow
        self.mlflow_tracker = MLflowTracker(experiment_name=experiment_name) if use_mlflow else None
        self.training_history = []
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_samples = 0
        
        for xb, yb in dataloader:
            xb = xb.to(self.device)
            yb = yb.to(self.device)
            
            pred = self.model(xb)
            if pred.dim() > 1 and pred.size(1) == 1:
                pred = pred.squeeze()
            
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * xb.size(0)
            n_samples += xb.size(0)
        
        return total_loss / n_samples if n_samples > 0 else 0.0
    
    def evaluate(
        self,
        dataloader: DataLoader,
        loss_fn: nn.Module
    ) -> Dict[str, float]:
        """Evaluate model."""
        self.model.eval()
        total_loss = 0.0
        n_samples = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for xb, yb in dataloader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                
                pred = self.model(xb)
                if pred.dim() > 1 and pred.size(1) == 1:
                    pred = pred.squeeze()
                
                loss = loss_fn(pred, yb)
                total_loss += loss.item() * xb.size(0)
                n_samples += xb.size(0)
                
                all_preds.append(pred.cpu().numpy())
                all_targets.append(yb.cpu().numpy())
        
        avg_loss = total_loss / n_samples if n_samples > 0 else 0.0
        
        # Calculate additional metrics
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        metrics = calculate_metrics(all_targets, all_preds)
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
        """Train model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            lr: Learning rate
            loss_fn: Loss function (defaults to MSE)
            optimizer: Optimizer (defaults to Adam)
            save_path: Path to save best model
            patience: Early stopping patience (None to disable)
        
        Returns:
            Training history dictionary
        """
        if loss_fn is None:
            loss_fn = nn.MSELoss()
        if optimizer is None:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'val_metrics': []}
        
        run = None
        if self.use_mlflow and self.mlflow_tracker:
            run = self.mlflow_tracker.start_run()
            self.mlflow_tracker.log_params({
                'epochs': epochs,
                'lr': lr,
                'model_type': type(self.model).__name__,
                'device': str(self.device)
            })
        
        try:
            for epoch in range(epochs):
                start_time = time.time()
                
                # Train
                train_loss = self.train_epoch(train_loader, loss_fn, optimizer)
                
                # Validate
                val_metrics = self.evaluate(val_loader, loss_fn)
                val_loss = val_metrics['loss']
                
                epoch_time = time.time() - start_time
                
                # Log metrics
                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                history['val_metrics'].append(val_metrics)
                
                if self.use_mlflow and self.mlflow_tracker:
                    self.mlflow_tracker.log_metrics({
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        **{f'val_{k}': v for k, v in val_metrics.items() if k != 'loss'},
                        'epoch_time': epoch_time
                    }, step=epoch)
                
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_loss:.6f}, "
                      f"Val Loss: {val_loss:.6f}, "
                      f"Val RMSE: {val_metrics.get('rmse', 0):.6f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    if save_path:
                        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
                        torch.save({
                            'model_state': self.model.state_dict(),
                            'epoch': epoch,
                            'val_loss': val_loss,
                            'val_metrics': val_metrics
                        }, save_path)
                else:
                    patience_counter += 1
                    if patience and patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            
            if self.use_mlflow and self.mlflow_tracker and save_path:
                self.mlflow_tracker.log_artifact(save_path)
        
        finally:
            if run:
                self.mlflow_tracker.end_run()
        
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
    """Convenience function to train a model."""
    trainer = Trainer(model, use_mlflow=use_mlflow)
    return trainer.train(train_loader, val_loader, epochs=epochs, lr=lr, save_path=save_path)

