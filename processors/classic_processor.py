"""
Classic OCR processors (Tesseract, EasyOCR, PyOCR).
"""
import os
import uuid
import threading
import time
import logging
import io
from typing import Dict, Any, Optional
from pdf2image import convert_from_bytes
import PIL.Image

import config
from processors.base_processor import BaseTask, BaseProcessor

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


class ClassicTask(BaseTask):
    """Task for classic OCR processing."""

    def __init__(self, task_id: str, filename: str, engine: str):
        super().__init__(task_id, filename, engine)
        self.engine = engine


class ClassicProcessor(BaseProcessor):
    """
    Processor for classic OCR engines (Tesseract, EasyOCR, PyOCR).
    These are rule-based OCR engines that don't use LLMs.
    """

    ENGINE_MODELS = {
        'tesseract': {
            'languages': ['rus+eng', 'eng'],
        },
        'easyocr': {
            'languages': ['ru', 'en'],
        },
        'pyocr': {
            'languages': ['rus+eng', 'eng'],
        }
    }

    def __init__(self):
        super().__init__()
        self.easyocr_reader = None
        self._init_readers()

    def _init_readers(self) -> None:
        """Initialize lazy-loaded OCR readers."""
        try:
            if easyocr and self.easyocr_reader is None:
                langs = self.ENGINE_MODELS['easyocr']['languages']
                self.easyocr_reader = easyocr.Reader(langs)
                logger.info("EasyOCR reader initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize EasyOCR: {e}")

    def submit_job(
        self,
        filename: str,
        content: bytes,
        engine: str = 'tesseract',
        detect_seal: bool = False,
        **kwargs
    ) -> Dict[str, str]:
        """
        Submit a new OCR job.

        Args:
            filename: Original filename
            content: File content bytes
            engine: OCR engine to use ('tesseract', 'easyocr', 'pyocr')
            detect_seal: Whether to detect seals
            **kwargs: Additional arguments

        Returns:
            Dict with job_id
        """
        task_id = str(uuid.uuid4())
        task = ClassicTask(task_id, filename, engine)
        task.detect_seal = detect_seal
        self.tasks[task_id] = task

        thread = threading.Thread(
            target=self._process_task,
            args=(task_id, content)
        )
        thread.start()

        return {'job_id': task_id}

    def _process_task(self, task_id: str, content: bytes) -> None:
        """Process OCR task with classic engine."""
        task = self.tasks[task_id]
        task.status = 'processing'
        task.updated_at = time.time()

        try:
            # Convert file to images
            images = []
            if task.filename.lower().endswith('.pdf'):
                images = convert_from_bytes(content)
            else:
                images = [PIL.Image.open(io.BytesIO(content))]
            
            # Optimize all images before processing
            images = [optimize_image(img) for img in images]
            
            task.total_pages = len(images)

            # Start seal detection in a separate thread if enabled
            if task.detect_seal:
                seal_thread = threading.Thread(
                    target=self._run_seal_detection,
                    args=(task_id, images)
                )
                seal_thread.start()

            pages_results = []

            for i, img in enumerate(images):
                # Store image
                img_io = io.BytesIO()
                img.save(img_io, 'PNG')
                img_io.seek(0)
                task.images[i + 1] = img_io.getvalue()

                # Save debug image
                try:
                    debug_filename = f"{task_id}_page_{i+1}.png"
                    debug_path = os.path.join(config.DEBUG_IMAGES_FOLDER, debug_filename)
                    with open(debug_path, "wb") as f:
                        f.write(task.images[i + 1])
                    logger.info(f"Saved debug image to {debug_path}")
                except Exception as e:
                    logger.warning(f"Failed to save debug image: {e}")

                # Perform OCR based on engine
                text = self._ocr_image(img, task.engine)
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

    def _ocr_image(self, img: PIL.Image.Image, engine: str) -> str:
        """Perform OCR on image with specified engine."""
        if engine == 'tesseract':
            return self._tesseract_ocr(img)
        elif engine == 'easyocr':
            return self._easyocr_ocr(img)
        elif engine == 'pyocr':
            return self._pyocr_ocr(img)
        else:
            return f"Unknown engine: {engine}"

    def _tesseract_ocr(self, img: PIL.Image.Image) -> str:
        """Perform OCR using Tesseract."""
        if not pytesseract:
            return "Tesseract not installed"

        lang = self.ENGINE_MODELS['tesseract']['languages'][0]
        return pytesseract.image_to_string(img, lang=lang)

    def _easyocr_ocr(self, img: PIL.Image.Image) -> str:
        """Perform OCR using EasyOCR."""
        if not self.easyocr_reader:
            return "EasyOCR not initialized"

        results = self.easyocr_reader.readtext(img)
        return "\n".join([res[1] for res in results])

    def _pyocr_ocr(self, img: PIL.Image.Image) -> str:
        """Perform OCR using PyOCR."""
        if not pyocr:
            return "PyOCR not installed"

        tools = pyocr.get_available_tools()
        if len(tools) == 0:
            return "No PyOCR tools available"

        tool = tools[0]
        lang = self.ENGINE_MODELS['pyocr']['languages'][0]

        return tool.image_to_string(
            img,
            lang=lang,
            builder=pyocr.builders.TextBuilder()
        )
