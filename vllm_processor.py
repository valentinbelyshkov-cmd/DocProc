import os
import uuid
import threading
import time
import logging
import PIL.Image
import io
from pdf2image import convert_from_path, convert_from_bytes
from vllm_models import GLMOCRModel, OpenRouterModel, OllamaModel, NoctrixLightOnOCRModel

logger = logging.getLogger(__name__)

MAX_IMAGE_DIMENSION = 1920
MAX_IMAGE_SIZE_KB = 300


def optimize_image(img: PIL.Image.Image) -> PIL.Image.Image:
    """Optimize image to fit within max dimension and file size limits."""
    width, height = img.size
    
    # Calculate scaling factor to fit within max dimension
    max_dim = max(width, height)
    if max_dim > MAX_IMAGE_DIMENSION:
        scale = MAX_IMAGE_DIMENSION / max_dim
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = img.resize((new_width, new_height), PIL.Image.Resampling.LANCZOS)
    
    # Compress to meet file size requirement
    output = io.BytesIO()
    img.save(output, format='JPEG', quality=85, optimize=True)
    
    while output.tell() > MAX_IMAGE_SIZE_KB * 1024 and output.tell() > 10240:
        output.seek(0)
        output.truncate()
        current_quality = getattr(img, '_last_quality', 85)
        new_quality = max(current_quality - 10, 30)
        img.save(output, format='JPEG', quality=new_quality, optimize=True)
    
    output.seek(0)
    return PIL.Image.open(output)

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
        logger.info(f"Starting processing task {task_id} for file {task.filename} using model {task.model_name}")
        
        try:
            # Determine if it's a PDF or image
            images = []
            if task.filename.lower().endswith('.pdf'):
                logger.info(f"Converting PDF {task.filename} to images")
                images = convert_from_bytes(content)
            else:
                logger.info(f"Opening image {task.filename}")
                images = [PIL.Image.open(io.BytesIO(content))]
            
            # Optimize all images before processing
            images = [optimize_image(img) for img in images]
            
            task.total_pages = len(images)
            if task.total_pages == 0:
                logger.warning(f"No pages found in file {task.filename}")
                raise ValueError(f"No pages could be extracted from {task.filename}")
                
            logger.info(f"Total pages to process: {task.total_pages}")
            model = self.get_model(task.model_name)
            
            pages_results = []
            for i, img in enumerate(images):
                logger.info(f"Processing page {i+1}/{task.total_pages}")
                # Store image
                img_io = io.BytesIO()
                img.save(img_io, 'PNG')
                img_io.seek(0)
                task.images[i+1] = img_io.getvalue()

                # Extract tables/text using VLLM
                logger.info(f"Calling model {task.model_name} for page {i+1}")
                result_data = model.extract_tables(img)
                logger.info(f"Model returned data for page {i+1}")
                
                # Build markdown from result
                markdown = f"## Results from page {i+1}\n\n"
                
                # Handle different result formats
                if isinstance(result_data, dict):
                    # If model returns both text and tables
                    if 'text' in result_data and result_data['text']:
                        markdown += result_data['text'] + "\n\n"
                    
                    if 'tables' in result_data and result_data['tables']:
                        markdown += "### Extracted Tables\n\n"
                        for table in result_data['tables']:
                            if isinstance(table, list):
                                for row in table:
                                    if isinstance(row, list):
                                        markdown += "| " + " | ".join([str(c) for c in row]) + " |\n"
                                    else:
                                        markdown += f"| {row} |\n"
                                markdown += "\n"
                elif isinstance(result_data, list):
                    # Assume it's a list of tables
                    for table in result_data:
                        if isinstance(table, list):
                            for row in table:
                                if isinstance(row, list):
                                    markdown += "| " + " | ".join([str(c) for c in row]) + " |\n"
                                else:
                                    markdown += f"| {row} |\n"
                            markdown += "\n"
                        else:
                            markdown += str(table) + "\n\n"
                else:
                    markdown += str(result_data)
                
                pages_results.append({
                    'page_num': i + 1,
                    'markdown': markdown,
                    'result_data': result_data, 
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
            logger.info(f"Task {task_id} completed successfully")
            
        except Exception as e:
            logger.exception(f"Error processing task {task_id}: {e}")
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