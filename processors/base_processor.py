"""
Base processor classes for OCR tasks.
Provides common interface and utilities for all processors.
"""
import uuid
import time
import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import PIL.Image
import io

import config
from models.seal_detector import get_seal_detector

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
        self.seals: List[Dict[str, Any]] = []

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
            'detect_seal': self.detect_seal,
            'seals_count': len(self.seals)
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

        return task.to_dict()

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

    def _run_seal_detection(self, task_id: str, images: List[PIL.Image.Image]) -> None:
        """Run seal detection on all pages in a separate thread."""
        task = self.tasks.get(task_id)
        if not task:
            return

        logger.info(f"Starting seal detection for task {task_id}")
        task.seals = []
        
        # Create task-specific seal folder
        task_seal_dir = os.path.join(config.SEALS_FOLDER, task_id)
        os.makedirs(task_seal_dir, exist_ok=True)

        try:
            detector = get_seal_detector(
                detector_type=config.SEAL_DETECTOR_TYPE,
                model_path=config.SEAL_MODEL_PATH
            )
            
            if not detector.is_available():
                logger.warning("Seal detector not available, skipping detection")
                return

            for i, img in enumerate(images):
                page_num = i + 1
                logger.debug(f"Detecting seals on page {page_num}")
                
                results = detector.detect(img)
                
                for j, r in enumerate(results):
                    # Crop seal
                    x1, y1, x2, y2 = r.bbox
                    # Ensure bbox is within image bounds
                    w, h = img.size
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    if x2 > x1 and y2 > y1:
                        seal_img = img.crop((x1, y1, x2, y2))
                        seal_filename = f"page_{page_num}_seal_{j+1}.png"
                        seal_path = os.path.join(task_seal_dir, seal_filename)
                        seal_img.save(seal_path)
                        
                        task.seals.append({
                            'page_num': page_num,
                            'bbox': r.bbox,
                            'confidence': float(r.confidence),
                            'seal_type': r.seal_type,
                            'filename': seal_filename
                        })
            
            logger.info(f"Seal detection completed for task {task_id}, found {len(task.seals)} seals")
        except Exception as e:
            logger.error(f"Error during seal detection for task {task_id}: {e}")
        
        task.updated_at = time.time()
