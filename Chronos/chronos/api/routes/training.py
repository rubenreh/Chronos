"""
Training management endpoints with background workers.

This module exposes three endpoints for managing long-running model training
jobs without blocking the API:

    - POST /training/submit   — accepts training parameters, spawns a background
      subprocess to run train_comprehensive.py, and returns a task ID immediately
    - GET  /training/status/{task_id} — polls the status of a submitted task
    - GET  /training/tasks    — lists all tasks (pending, running, completed, failed)

These endpoints delegate to the BackgroundWorker singleton defined in workers.py,
which manages an in-memory task registry and spawns asyncio subprocesses.

Workflow:
    1. Client POSTs to /training/submit with data_path, model_type, epochs, etc.
    2. The handler generates a UUID task_id and tells the worker to run training
    3. Client polls GET /training/status/{task_id} until status is COMPLETED or FAILED
    4. On COMPLETED, the trained .pth checkpoint is available in model_dir
"""
import uuid                                     # Generate unique task identifiers
from fastapi import APIRouter, HTTPException, BackgroundTasks  # Router, error handling, background tasks
from pydantic import BaseModel, Field           # Pydantic models for request/response validation
from typing import Optional, Dict, Any          # Type hints

from chronos.api.workers import get_worker, TaskStatus  # Singleton worker and status enum

# Create router under /training prefix, grouped as "training" in OpenAPI docs
router = APIRouter(prefix="/training", tags=["training"])

# Retrieve the global BackgroundWorker singleton that manages subprocess tasks
WORKER = get_worker()


# ---------------------------------------------------------------------------
# Request / Response schemas (local to this module)
# ---------------------------------------------------------------------------

class TrainingRequest(BaseModel):
    """Request body for POST /training/submit.

    Specifies all parameters needed to launch a training run: which data
    to use, which architecture, how many epochs, where to save the model,
    and whether to track the experiment with MLflow.
    """
    data_path: str = Field(..., description="Path to training data CSV")                         # Required: input data location
    model_type: str = Field("lstm", description="Model type: lstm, tcn, transformer")            # Architecture selection
    epochs: int = Field(20, ge=1, le=1000, description="Number of training epochs")              # Training duration (clamped 1–1000)
    model_dir: str = Field("artifacts", description="Directory to save models")                  # Output directory for .pth files
    use_mlflow: bool = Field(True, description="Whether to use MLflow tracking")                 # Experiment tracking toggle
    experiment_name: Optional[str] = Field(None, description="MLflow experiment name")           # Optional custom experiment name


class TrainingResponse(BaseModel):
    """Response body returned by POST /training/submit.

    Confirms the task was accepted and provides the task_id for status polling.
    """
    task_id: str           # UUID identifying the submitted task
    status: str            # Initial status (always "pending" at submission time)
    message: str           # Human-readable confirmation message


class TaskStatusResponse(BaseModel):
    """Response body for GET /training/status/{task_id}.

    Contains the full lifecycle metadata of a training task: when it was
    created, started, completed, and any output or error messages.
    """
    task_id: str                                 # UUID of the task
    status: str                                  # Current state: pending, running, completed, failed
    created_at: str                              # ISO-8601 timestamp when the task was submitted
    started_at: Optional[str] = None             # ISO-8601 timestamp when execution began (None if still pending)
    completed_at: Optional[str] = None           # ISO-8601 timestamp when execution finished
    task_type: str                               # Always "training" for this module
    output: Optional[str] = None                 # Captured stdout on success
    error: Optional[str] = None                  # Captured stderr on failure


# ---------------------------------------------------------------------------
# Endpoint handlers
# ---------------------------------------------------------------------------

@router.post("/submit", response_model=TrainingResponse)
async def submit_training(req: TrainingRequest):
    """Submit a training task to run in the background.

    Generates a unique task ID, registers the task with the background worker,
    and returns immediately. The actual training runs as an asyncio subprocess.
    Use GET /training/status/{task_id} to poll for completion.
    """
    # Generate a UUID4 to uniquely identify this training run
    task_id = str(uuid.uuid4())

    try:
        # Delegate to the BackgroundWorker, which registers the task and
        # schedules an asyncio coroutine to spawn the training subprocess
        WORKER.submit_training_task(
            task_id=task_id,                                     # Unique identifier
            data_path=req.data_path,                             # Path to the CSV training data
            model_type=req.model_type,                           # Architecture: lstm / tcn / transformer
            epochs=req.epochs,                                   # Number of training epochs
            model_dir=req.model_dir,                             # Where to save the trained .pth checkpoint
            use_mlflow=req.use_mlflow                            # Whether to log to MLflow
        )

        return TrainingResponse(
            task_id=task_id,                                     # Return the task ID for later polling
            status="pending",                                    # Task starts in PENDING state
            message=f"Training task submitted. Use task_id to check status."
        )
    except Exception as e:
        # If submission itself fails (e.g. event-loop issue), return 500
        raise HTTPException(status_code=500, detail=f"Failed to submit training task: {str(e)}")


@router.get("/status/{task_id}", response_model=TaskStatusResponse)
async def get_training_status(task_id: str):
    """Retrieve the current status of a previously submitted training task.

    Returns the full task metadata including timestamps, stdout output
    (on success), and stderr error messages (on failure).
    """
    task = WORKER.get_task_status(task_id)                       # Look up the task in the in-memory registry

    if task is None:                                             # Unknown task_id → 404
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    # Map the internal task dict to the Pydantic response model
    return TaskStatusResponse(
        task_id=task_id,
        status=task['status'],                                   # Current lifecycle state
        created_at=task.get('created_at', ''),                   # When the task was first registered
        started_at=task.get('started_at'),                       # When subprocess execution began
        completed_at=task.get('completed_at'),                   # When subprocess finished
        task_type=task.get('task_type', 'training'),             # Task category
        output=task.get('output'),                               # Stdout capture (populated on success)
        error=task.get('error')                                  # Stderr capture (populated on failure)
    )


@router.get("/tasks", response_model=Dict[str, Any])
async def list_tasks():
    """List all training tasks (pending, running, completed, and failed).

    Returns the full task registry and a count, useful for dashboard UIs
    that need to display all historical and active training runs.
    """
    tasks = WORKER.list_tasks()                                  # Shallow copy of the entire task registry
    return {"tasks": tasks, "count": len(tasks)}                 # Return tasks dict + total count
