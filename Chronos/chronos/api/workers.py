"""
Background workers for training and long-running tasks.

This module manages asynchronous background training jobs so that the API can
accept a training request, return immediately with a task ID, and let the
actual training run as a subprocess without blocking the FastAPI event loop.

Architecture:
    - BackgroundWorker maintains an in-memory registry (dict) of submitted tasks
    - Each task runs ``python -m chronos.training.train_comprehensive`` as an
      asyncio subprocess, capturing stdout/stderr
    - Task lifecycle: PENDING → RUNNING → COMPLETED | FAILED
    - The training route module (routes/training.py) exposes HTTP endpoints
      that delegate to the singleton BackgroundWorker instance

Production note:
    In a production deployment this would be replaced by a proper distributed
    task queue (Celery + Redis, or AWS SQS) for persistence and scalability.
    The in-memory approach here is suitable for single-server development.
"""
import asyncio                                  # Async subprocess execution and event-loop access
import os                                       # File-system utilities (available for path operations)
import subprocess                               # (Available as fallback; asyncio.create_subprocess_exec is preferred)
from typing import Dict, Optional               # Type hints for task registry
from datetime import datetime                   # Timestamping task lifecycle events
from enum import Enum                           # Enumeration for task states


class TaskStatus(Enum):
    """Enumeration of possible states for a background task.

    The lifecycle is: PENDING → RUNNING → COMPLETED or FAILED.
    """
    PENDING = "pending"        # Task has been accepted but not yet started
    RUNNING = "running"        # Subprocess is actively executing
    COMPLETED = "completed"    # Subprocess exited with return code 0
    FAILED = "failed"          # Subprocess exited with a non-zero code or raised an exception


class BackgroundWorker:
    """Manages background training tasks via asyncio subprocesses.

    Provides methods to submit new training jobs, poll their status, and
    list all tasks. Each task's metadata (status, timestamps, output, errors)
    is stored in self.tasks keyed by a UUID task ID.
    """

    def __init__(self):
        """Initialize the worker with an empty task registry."""
        # In-memory dict: task_id (str) → task metadata (dict)
        self.tasks: Dict[str, Dict] = {}

    def submit_training_task(
        self,
        task_id: str,
        data_path: str,
        model_type: str = "lstm",
        epochs: int = 20,
        model_dir: str = "artifacts",
        use_mlflow: bool = True
    ) -> str:
        """Submit a training task to run in the background.

        Registers the task as PENDING, then schedules an asyncio coroutine
        that will spawn a subprocess to execute the comprehensive training
        script. Returns the task_id immediately so the HTTP handler can
        respond without waiting for training to finish.

        Args:
            task_id: Unique identifier (UUID) for this task
            data_path: Path to the CSV training data file
            model_type: Architecture to train ("lstm", "tcn", "transformer")
            epochs: Number of training epochs
            model_dir: Directory where the trained .pth checkpoint will be saved
            use_mlflow: Whether to log metrics/artifacts to MLflow

        Returns:
            The task_id string (echoed back for convenience)
        """
        # Register the task with initial PENDING status and all configuration
        self.tasks[task_id] = {
            'status': TaskStatus.PENDING.value,              # Start in "pending" state
            'created_at': datetime.now().isoformat(),        # ISO-8601 creation timestamp
            'task_type': 'training',                         # Distinguishes training from other future task types
            'data_path': data_path,                          # Path to input CSV
            'model_type': model_type,                        # Architecture selection
            'epochs': epochs,                                # Training duration
            'model_dir': model_dir,                          # Output directory for checkpoints
            'use_mlflow': use_mlflow,                        # MLflow tracking toggle
            'output': None,                                  # Will hold stdout on completion
            'error': None                                    # Will hold stderr on failure
        }

        # Schedule the async training coroutine on the running event loop.
        # In production, a proper task queue (Celery/RQ) would replace this.
        import asyncio
        try:
            loop = asyncio.get_event_loop()                  # Get the current event loop (FastAPI's uvloop)
            loop.create_task(self._run_training_task(task_id))  # Fire-and-forget coroutine
        except RuntimeError:
            # No running event loop (e.g. called from a sync context) —
            # fall back to asyncio.run which creates a temporary loop
            asyncio.run(self._run_training_task(task_id))

        return task_id

    async def _run_training_task(self, task_id: str):
        """Execute the training subprocess and update task status on completion.

        Builds the CLI command for train_comprehensive.py, spawns it as an
        async subprocess (non-blocking), captures stdout/stderr, and updates
        the task registry with the result.
        """
        if task_id not in self.tasks:                        # Guard against race conditions
            return

        task = self.tasks[task_id]
        task['status'] = TaskStatus.RUNNING.value            # Transition: PENDING → RUNNING
        task['started_at'] = datetime.now().isoformat()      # Record when execution actually began

        try:
            # Build the CLI command that invokes the comprehensive training script
            cmd = [
                'python', '-m', 'chronos.training.train_comprehensive',  # Module-style invocation
                '--data', task['data_path'],                             # Input data CSV path
                '--models', task['model_type'],                          # Which architecture(s) to train
                '--epochs', str(task['epochs']),                         # Number of epochs (must be string for CLI)
                '--model-dir', task['model_dir']                         # Where to save the .pth checkpoint
            ]

            if task['use_mlflow']:                           # Optionally enable MLflow experiment tracking
                cmd.append('--use-mlflow')

            # Spawn the training process asynchronously so it doesn't block
            # the FastAPI event loop. Pipes capture stdout and stderr.
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,              # Capture standard output
                stderr=asyncio.subprocess.PIPE               # Capture standard error
            )

            # Wait for the subprocess to finish and collect its output
            stdout, stderr = await process.communicate()

            if process.returncode == 0:                      # Exit code 0 = success
                task['status'] = TaskStatus.COMPLETED.value  # Transition: RUNNING → COMPLETED
                task['output'] = stdout.decode()              # Store decoded stdout for later retrieval
            else:
                task['status'] = TaskStatus.FAILED.value     # Non-zero exit = failure
                task['error'] = stderr.decode()              # Store decoded stderr for debugging

            task['completed_at'] = datetime.now().isoformat()  # Record completion timestamp

        except Exception as e:
            # Catch any unexpected errors (e.g. binary not found, permission denied)
            task['status'] = TaskStatus.FAILED.value
            task['error'] = str(e)                           # Store the exception message
            task['completed_at'] = datetime.now().isoformat()

    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """Look up the current status of a task by its ID.

        Args:
            task_id: The UUID assigned when the task was submitted

        Returns:
            The task metadata dict, or None if the task_id is unknown
        """
        return self.tasks.get(task_id)                       # dict.get returns None on miss

    def list_tasks(self) -> Dict[str, Dict]:
        """Return a shallow copy of all tasks in the registry.

        Returns:
            Dictionary mapping task_id → task metadata for every known task
        """
        return self.tasks.copy()                             # Copy prevents external mutation of internal state


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
# A single BackgroundWorker instance is shared across the application.
# The training route module accesses it via get_worker().
_worker = BackgroundWorker()


def get_worker() -> BackgroundWorker:
    """Return the global BackgroundWorker singleton.

    Called by the training route handlers to submit tasks and query status.
    """
    return _worker
