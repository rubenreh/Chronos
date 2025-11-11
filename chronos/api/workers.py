"""Background workers for training and long-running tasks."""
import asyncio
import os
import subprocess
from typing import Dict, Optional
from datetime import datetime
from enum import Enum


class TaskStatus(Enum):
    """Task status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class BackgroundWorker:
    """Background worker for running training tasks."""
    
    def __init__(self):
        """Initialize background worker."""
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
        """Submit a training task to run in background.
        
        Args:
            task_id: Unique task identifier
            data_path: Path to training data
            model_type: Type of model to train
            epochs: Number of training epochs
            model_dir: Directory to save models
            use_mlflow: Whether to use MLflow tracking
        
        Returns:
            Task ID
        """
        self.tasks[task_id] = {
            'status': TaskStatus.PENDING.value,
            'created_at': datetime.now().isoformat(),
            'task_type': 'training',
            'data_path': data_path,
            'model_type': model_type,
            'epochs': epochs,
            'model_dir': model_dir,
            'use_mlflow': use_mlflow,
            'output': None,
            'error': None
        }
        
        # Note: Task will be started by the event loop
        # In production, use a proper task queue like Celery or RQ
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(self._run_training_task(task_id))
        except RuntimeError:
            # If no event loop, create one
            asyncio.run(self._run_training_task(task_id))
        
        return task_id
    
    async def _run_training_task(self, task_id: str):
        """Run training task in background."""
        if task_id not in self.tasks:
            return
        
        task = self.tasks[task_id]
        task['status'] = TaskStatus.RUNNING.value
        task['started_at'] = datetime.now().isoformat()
        
        try:
            # Build command
            cmd = [
                'python', '-m', 'chronos.training.train_comprehensive',
                '--data', task['data_path'],
                '--models', task['model_type'],
                '--epochs', str(task['epochs']),
                '--model-dir', task['model_dir']
            ]
            
            if task['use_mlflow']:
                cmd.append('--use-mlflow')
            
            # Run training
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                task['status'] = TaskStatus.COMPLETED.value
                task['output'] = stdout.decode()
            else:
                task['status'] = TaskStatus.FAILED.value
                task['error'] = stderr.decode()
            
            task['completed_at'] = datetime.now().isoformat()
            
        except Exception as e:
            task['status'] = TaskStatus.FAILED.value
            task['error'] = str(e)
            task['completed_at'] = datetime.now().isoformat()
    
    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """Get status of a task.
        
        Args:
            task_id: Task identifier
        
        Returns:
            Task status dictionary or None if not found
        """
        return self.tasks.get(task_id)
    
    def list_tasks(self) -> Dict[str, Dict]:
        """List all tasks.
        
        Returns:
            Dictionary of task_id -> task_info
        """
        return self.tasks.copy()


# Global worker instance
_worker = BackgroundWorker()


def get_worker() -> BackgroundWorker:
    """Get global background worker instance."""
    return _worker

