"""
MLflow Tracker – Experiment Tracking for Chronos
==================================================
This module provides a thin abstraction over the MLflow Python SDK so that the
rest of the Chronos codebase never imports or calls mlflow directly. Benefits:

  • Single point of change if the tracking backend is swapped (e.g., to W&B).
  • Default tracking URI points to a local ./mlruns directory — no server needed
    for development. In production the URI can point to a remote MLflow server.
  • Exposes only the five operations Chronos actually uses:
        start_run  → begin an experiment run
        log_params → record hyperparameters (once per run)
        log_metrics → record epoch-level metrics (train loss, val loss, etc.)
        log_model  → persist a serialized PyTorch model as an artifact
        log_artifact → persist any file (checkpoint, plot) as an artifact
        end_run    → close the run cleanly

Typical call sequence (inside Trainer.train()):
    tracker = MLflowTracker("chronos_forecasting")
    tracker.start_run()
    tracker.log_params({"lr": 1e-3, "epochs": 20})
    for epoch in range(20):
        tracker.log_metrics({"train_loss": ..., "val_loss": ...}, step=epoch)
    tracker.log_artifact("artifacts/best_model.pth")
    tracker.end_run()
"""

import os
import mlflow            # Core MLflow API for tracking runs, params, metrics
import mlflow.pytorch    # PyTorch-specific model logging (serialization)
from typing import Dict, Optional, Any
from pathlib import Path


class MLflowTracker:
    """Wrapper around MLflow's tracking API, scoped to a single experiment.

    One MLflowTracker instance maps to one MLflow *experiment* (a named
    collection of runs). Each call to start_run() creates a new *run* inside
    that experiment.
    """

    def __init__(
        self,
        experiment_name: str = "chronos_forecasting",
        tracking_uri: Optional[str] = None
    ):
        """Initialize MLflow tracker and configure the experiment.

        Args:
            experiment_name: Logical name grouping related training runs together
                             (e.g., "chronos_forecasting" or "lstm_tuning").
            tracking_uri: URI telling MLflow where to store run data. When None,
                          defaults to a local file-based store at ./mlruns so no
                          external server is required during development.
        """
        # Fall back to a local file URI when no remote tracking server is given.
        if tracking_uri is None:
            tracking_uri = f"file://{Path.cwd()}/mlruns"

        # Point the MLflow client at the chosen backend (local or remote).
        mlflow.set_tracking_uri(tracking_uri)

        # Create the experiment if it doesn't exist yet, or select it if it does.
        mlflow.set_experiment(experiment_name)

        # Store the name so callers can inspect which experiment this tracker uses.
        self.experiment_name = experiment_name

    def start_run(self, run_name: Optional[str] = None) -> mlflow.ActiveRun:
        """Start a new MLflow run under the configured experiment.

        Args:
            run_name: Optional human-readable label for this run (shown in the
                      MLflow UI). If None, MLflow auto-generates a name.

        Returns:
            An ActiveRun context manager representing the newly started run.
        """
        return mlflow.start_run(run_name=run_name)

    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters (called once at the start of a run).

        Args:
            params: Dictionary of parameter names to values, e.g.
                    {"lr": 1e-3, "epochs": 20, "model_type": "LSTM"}.
        """
        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log numeric metrics (typically called once per epoch).

        Args:
            metrics: Dictionary of metric names to float values, e.g.
                     {"train_loss": 0.032, "val_loss": 0.041}.
            step: The training step (usually the epoch number) so that MLflow
                  can plot metrics over time.
        """
        mlflow.log_metrics(metrics, step=step)

    def log_model(self, model: Any, artifact_path: str = "model"):
        """Serialize and log a PyTorch model as an MLflow artifact.

        The model is saved using MLflow's built-in PyTorch flavor, which stores
        it in a format that can be loaded back with mlflow.pytorch.load_model().

        Args:
            model: A PyTorch nn.Module to persist.
            artifact_path: Sub-directory name inside the run's artifact store.
        """
        mlflow.pytorch.log_model(model, artifact_path)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Upload any local file (checkpoint, CSV, plot) as a run artifact.

        Args:
            local_path: Path to the file on the local filesystem.
            artifact_path: Optional sub-directory inside the artifact store.
        """
        mlflow.log_artifact(local_path, artifact_path)

    def end_run(self):
        """End the currently active MLflow run.

        This must be called after training completes (or in a finally block) to
        ensure the run is marked as "FINISHED" rather than left in "RUNNING".
        """
        mlflow.end_run()


def setup_mlflow(
    experiment_name: str = "chronos_forecasting",
    tracking_uri: Optional[str] = None
) -> MLflowTracker:
    """Factory function that creates and returns a configured MLflowTracker.

    This is a convenience shortcut so callers don't need to import the class
    directly:
        tracker = setup_mlflow("my_experiment")

    Args:
        experiment_name: Name of the MLflow experiment.
        tracking_uri: Optional URI for a remote tracking server.

    Returns:
        A ready-to-use MLflowTracker instance.
    """
    return MLflowTracker(experiment_name=experiment_name, tracking_uri=tracking_uri)
