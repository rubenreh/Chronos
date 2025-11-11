"""Training endpoints with background workers."""
import uuid
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

from chronos.api.workers import get_worker, TaskStatus

router = APIRouter(prefix="/training", tags=["training"])

WORKER = get_worker()


class TrainingRequest(BaseModel):
    """Request for training a model."""
    data_path: str = Field(..., description="Path to training data CSV")
    model_type: str = Field("lstm", description="Model type: lstm, tcn, transformer")
    epochs: int = Field(20, ge=1, le=1000, description="Number of training epochs")
    model_dir: str = Field("artifacts", description="Directory to save models")
    use_mlflow: bool = Field(True, description="Whether to use MLflow tracking")
    experiment_name: Optional[str] = Field(None, description="MLflow experiment name")


class TrainingResponse(BaseModel):
    """Response from training submission."""
    task_id: str
    status: str
    message: str


class TaskStatusResponse(BaseModel):
    """Response for task status."""
    task_id: str
    status: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    task_type: str
    output: Optional[str] = None
    error: Optional[str] = None


@router.post("/submit", response_model=TrainingResponse)
async def submit_training(req: TrainingRequest):
    """Submit a training task to run in background.
    
    Returns immediately with task ID. Use /training/status/{task_id} to check progress.
    """
    # Generate unique task ID
    task_id = str(uuid.uuid4())
    
    # Submit task
    try:
        WORKER.submit_training_task(
            task_id=task_id,
            data_path=req.data_path,
            model_type=req.model_type,
            epochs=req.epochs,
            model_dir=req.model_dir,
            use_mlflow=req.use_mlflow
        )
        
        return TrainingResponse(
            task_id=task_id,
            status="pending",
            message=f"Training task submitted. Use task_id to check status."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit training task: {str(e)}")


@router.get("/status/{task_id}", response_model=TaskStatusResponse)
async def get_training_status(task_id: str):
    """Get status of a training task."""
    task = WORKER.get_task_status(task_id)
    
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    return TaskStatusResponse(
        task_id=task_id,
        status=task['status'],
        created_at=task.get('created_at', ''),
        started_at=task.get('started_at'),
        completed_at=task.get('completed_at'),
        task_type=task.get('task_type', 'training'),
        output=task.get('output'),
        error=task.get('error')
    )


@router.get("/tasks", response_model=Dict[str, Any])
async def list_tasks():
    """List all training tasks."""
    tasks = WORKER.list_tasks()
    return {"tasks": tasks, "count": len(tasks)}

