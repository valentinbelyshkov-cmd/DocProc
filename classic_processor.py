import os
import uuid
import threading
import time
import logging
import PIL.Image
import io
from pdf2image import convert_from_bytes

logger = logging.getLogger(__name__)

try:
    import pytesseract
except ImportError:
    pytesseract = None

try:
    import easyocr
except ImportError:
    easyocr = None

try:
    import pyocr
    import pyocr.builders
except ImportError:
    pyocr = None

class ClassicTask:
    def __init__(self, task_id, filename, engine):
        self.task_id = task_id
        self.filename = filename
        self.engine = engine
        self.status = 'queued'
        self.progress = 0
        self.total_pages = 0
        self.processed_pages = 0
        self.result = None
        self.error = None
        self.created_at = time.time()
        self.updated_at = time.time()
        self.images = {}

class ClassicProcessor:
    def __init__(self):
        self.tasks = {}
        self.easyocr_reader = None

    def submit_job(self, filename, content, engine='tesseract'):
        task_id = str(uuid.uuid4())
        task = ClassicTask(task_id, filename, engine)
        self.tasks[task_id] = task
        
        thread = threading.Thread(target=self._process_task, args=(task_id, content))
        thread.start()
        
        return {'job_id': task_id}

    def _process_task(self, task_id, content):
        task = self.tasks[task_id]
        task.status = 'processing'
        task.updated_at = time.time()
        
        try:
            images = []
            if task.filename.lower().endswith('.pdf'):
                images = convert_from_bytes(content)
            else:
                images = [PIL.Image.open(io.BytesIO(content))]
            
            task.total_pages = len(images)
            
            pages_results = []
            
            if task.engine == 'easyocr' and self.easyocr_reader is None:
                if easyocr:
                    self.easyocr_reader = easyocr.Reader(['ru', 'en'])
            
            for i, img in enumerate(images):
                img_io = io.BytesIO()
                img.save(img_io, 'PNG')
                img_io.seek(0)
                task.images[i+1] = img_io.getvalue()

                text = ""
                if task.engine == 'tesseract':
                    if pytesseract:
                        text = pytesseract.image_to_string(img, lang='rus+eng')
                    else:
                        text = "Tesseract not installed"
                elif task.engine == 'easyocr':
                    if self.easyocr_reader:
                        results = self.easyocr_reader.readtext(img)
                        text = "\n".join([res[1] for res in results])
                    else:
                        text = "EasyOCR not installed"
                elif task.engine == 'pyocr':
                    if pyocr:
                        tools = pyocr.get_available_tools()
                        if len(tools) > 0:
                            tool = tools[0]
                            text = tool.image_to_string(
                                img,
                                lang="rus+eng",
                                builder=pyocr.builders.TextBuilder()
                            )
                        else:
                            text = "No PyOCR tools available"
                    else:
                        text = "PyOCR not installed"
                
                pages_results.append({
                    'page_num': i + 1,
                    'markdown': text,
                    'result_json': []
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
