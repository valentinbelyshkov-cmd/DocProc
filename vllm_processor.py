import os
import uuid
import threading
import time
import logging
import PIL.Image
import io
from pdf2image import convert_from_path, convert_from_bytes
from vllm_models import GLMOCRModel, PaddleOCRVLModel, OpenRouterModel, OllamaModel, NoctrixLightOnOCRModel

logger = logging.getLogger(__name__)

class VLLMTask:
    def __init__(self, task_id, filename, model_name):
        self.task_id = task_id
        self.filename = filename
        self.model_name = model_name
        self.status = 'queued'
        self.progress = 0
        self.total_pages = 0
        self.processed_pages = 0
        self.result = None
        self.error = None
        self.created_at = time.time()
        self.updated_at = time.time()
        self.images = {} # Store images for retrieval

class VLLMProcessor:
    def __init__(self):
        self.tasks = {}
        self.models = {
            'glm-ocr': GLMOCRModel(),
            'paddle-vl': PaddleOCRVLModel(),
            'openrouter': OpenRouterModel(),
            'ollama-glm': OllamaModel(model_name='glm-ocr'),
            'ollama': OllamaModel(),
            'noctrix': NoctrixLightOnOCRModel()
        }

    def get_model(self, model_name):
        return self.models.get(model_name, self.models['openrouter'])

    def submit_job(self, filename, content, model_name='openrouter'):
        task_id = str(uuid.uuid4())
        task = VLLMTask(task_id, filename, model_name)
        self.tasks[task_id] = task
        
        thread = threading.Thread(target=self._process_task, args=(task_id, content))
        thread.start()
        
        return {'job_id': task_id}

    def _process_task(self, task_id, content):
        task = self.tasks[task_id]
        task.status = 'processing'
        task.updated_at = time.time()
        
        try:
            # Determine if it's a PDF or image
            images = []
            if task.filename.lower().endswith('.pdf'):
                images = convert_from_bytes(content)
            else:
                images = [PIL.Image.open(io.BytesIO(content))]
            
            task.total_pages = len(images)
            model = self.get_model(task.model_name)
            
            pages_results = []
            for i, img in enumerate(images):
                # Store image
                img_io = io.BytesIO()
                img.save(img_io, 'PNG')
                img_io.seek(0)
                task.images[i+1] = img_io.getvalue()

                # Extract tables using VLLM
                tables = model.extract_tables(img)
                
                # Mock markdown for now
                markdown = f"## Tables from page {i+1}\n\n"
                if tables:
                    for table in tables:
                        for row in table:
                            markdown += "| " + " | ".join([str(c) for c in row]) + " |\n"
                        markdown += "\n"
                
                pages_results.append({
                    'page_num': i + 1,
                    'markdown': markdown,
                    'tables': tables, # Original tables from VLLM
                    'result_json': [] # Compatible with existing code
                })
                
                task.processed_pages += 1
                task.progress = int((task.processed_pages / task.total_pages) * 100)
                task.updated_at = time.time()

            task.result = {
                'job_id': task_id,
                'status': 'completed',
                'pages': pages_results,
                'total_pages': task.total_pages,
                'processed_pages': task.processed_pages
            }
            task.status = 'completed'
            
        except Exception as e:
            logger.error(f"Error processing task {task_id}: {e}")
            task.status = 'failed'
            task.error = str(e)
        
        task.updated_at = time.time()

    def get_status(self, task_id):
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

    def get_result(self, task_id):
        task = self.tasks.get(task_id)
        if not task or task.status != 'completed':
            raise ValueError("Result not ready or task not found")
        return task.result

    def get_image(self, task_id, page_num):
        task = self.tasks.get(task_id)
        if not task or page_num not in task.images:
            return None
        return io.BytesIO(task.images[page_num])
