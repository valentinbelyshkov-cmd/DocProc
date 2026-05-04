"""
Base processor classes for OCR tasks.
Provides common interface and utilities for all processors.
"""
import uuid
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class BaseTask:
    """Base class for all OCR tasks."""

    def __init__(self, task_id: str, filename: str, model_name: str = None):
        self.task_id = task_id
        self.filename = filename
        self.model_name = model_name
        self.status = 'queued'
        self.progress = 0
        self.total_pages = 0
        self.processed_pages = 0
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.created_at = time.time()
        self.updated_at = time.time()
        self.images: Dict[int, bytes] = {}
        self.detect_seal = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary representation."""
        return {
            'task_id': self.task_id,
            'filename': self.filename,
            'model_name': self.model_name,
            'status': self.status,
            'progress': self.progress,
            'total_pages': self.total_pages,
            'processed_pages': self.processed_pages,
            'error': self.error,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'detect_seal': self.detect_seal
        }


class BaseProcessor(ABC):
    """
    Abstract base processor for OCR tasks.
    All processors should inherit from this class and implement required methods.
    """

    def __init__(self):
        self.tasks: Dict[str, BaseTask] = {}

    @abstractmethod
    def submit_job(self, filename: str, content: bytes, **kwargs) -> Dict[str, str]:
        """Submit a new OCR job. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _process_task(self, task_id: str, content: bytes) -> None:
        """Process the OCR task. Must be implemented by subclasses."""
        pass

    def get_status(self, task_id: str) -> Dict[str, Any]:
        """Get the status of a task."""
        task = self.tasks.get(task_id)
        if not task:
            raise ValueError("Task not found")

        return {
            'status': task.status,
            'progress': task.progress,
            'total_pages': task.total_pages,
            'processed_pages': task.processed_pages,
            'error': task.error,
            'created_at': task.created_at,
            'updated_at': task.updated_at
        }

    def get_result(self, task_id: str) -> Dict[str, Any]:
        """Get the result of a completed task."""
        task = self.tasks.get(task_id)
        if not task or task.status != 'completed':
            raise ValueError("Result not ready or task not found")
        return task.result

    def get_image(self, task_id: str, page_num: int) -> Optional[bytes]:
        """Get the image for a specific page."""
        task = self.tasks.get(task_id)
        if not task or page_num not in task.images:
            return None
        return self.tasks[task_id].images[page_num]

    def cleanup_expired_tasks(self, max_age_seconds: int = 3600) -> int:
        """Remove expired tasks to free memory."""
        current_time = time.time()
        expired = [
            task_id for task_id, task in self.tasks.items()
            if current_time - task.updated_at > max_age_seconds
        ]
        for task_id in expired:
            del self.tasks[task_id]
        return len(expired)
