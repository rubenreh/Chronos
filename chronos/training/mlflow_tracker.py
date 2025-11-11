"""MLflow integration for experiment tracking."""
import os
import mlflow
import mlflow.pytorch
from typing import Dict, Optional, Any
from pathlib import Path


class MLflowTracker:
    """MLflow experiment tracker."""
    
    def __init__(
        self,
        experiment_name: str = "chronos_forecasting",
        tracking_uri: Optional[str] = None
    ):
        """Initialize MLflow tracker.
        
        Args:
            experiment_name: Name of MLflow experiment
            tracking_uri: Optional tracking URI (defaults to local ./mlruns)
        """
        if tracking_uri is None:
            tracking_uri = f"file://{Path.cwd()}/mlruns"
        
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name
    
    def start_run(self, run_name: Optional[str] = None) -> mlflow.ActiveRun:
        """Start a new MLflow run."""
        return mlflow.start_run(run_name=run_name)
    
    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters."""
        mlflow.log_params(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics."""
        mlflow.log_metrics(metrics, step=step)
    
    def log_model(self, model: Any, artifact_path: str = "model"):
        """Log PyTorch model."""
        mlflow.pytorch.log_model(model, artifact_path)
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log an artifact."""
        mlflow.log_artifact(local_path, artifact_path)
    
    def end_run(self):
        """End current run."""
        mlflow.end_run()


def setup_mlflow(
    experiment_name: str = "chronos_forecasting",
    tracking_uri: Optional[str] = None
) -> MLflowTracker:
    """Convenience function to setup MLflow."""
    return MLflowTracker(experiment_name=experiment_name, tracking_uri=tracking_uri)

