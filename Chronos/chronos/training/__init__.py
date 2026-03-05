"""
Training Package – Chronos Time-Series Forecasting System
=========================================================
This package exposes the core training infrastructure used by Chronos:

  • Trainer      – orchestrates the PyTorch training loop with early stopping,
                   validation, checkpointing, and optional MLflow metric logging.
  • train_model  – a convenience wrapper around Trainer for quick one-call training.
  • MLflowTracker – thin wrapper over the MLflow Python API that manages
                    experiment runs, hyperparameter logging, metric logging, and
                    artifact storage (model checkpoints, plots, etc.).
  • setup_mlflow – factory function that creates a configured MLflowTracker.

Typical usage in the project:
    from chronos.training import Trainer, MLflowTracker
    tracker = MLflowTracker(experiment_name="my_experiment")
    trainer = Trainer(model, use_mlflow=True)
    history = trainer.train(train_loader, val_loader, epochs=20)
"""

# Import the Trainer class and its convenience function from the trainer module.
# Trainer handles the full training loop (epoch iteration, loss computation,
# early stopping, checkpoint saving, MLflow logging).
from .trainer import Trainer, train_model

# Import MLflow integration helpers. MLflowTracker wraps the MLflow SDK so
# the rest of the codebase never calls mlflow directly.
from .mlflow_tracker import MLflowTracker, setup_mlflow

# __all__ controls what is exported when another module does
# `from chronos.training import *`. Listing symbols here keeps the public
# API explicit and tidy.
__all__ = ['Trainer', 'train_model', 'MLflowTracker', 'setup_mlflow']
