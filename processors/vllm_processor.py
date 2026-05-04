"""
VLLM Processor for OCR tasks using vision-capable models.
"""
import os
import uuid
import threading
import logging
import io
import time
import re
from typing import Dict, Any, Optional, List
from pdf2image import convert_from_bytes
import PIL.Image

from processors.base_processor import BaseTask, BaseProcessor
from models.models_registry import ModelRegistry
from models.base_model import ModelConfig
from processors.handlers_registry import DocumentHandlerRegistry
from models.seal_detector import get_seal_detector

logger = logging.getLogger(__name__)


class VLLMTask(BaseTask):
    """Task for VLLM-based OCR processing."""

    def __init__(self, task_id: str, filename: str, model_name: str):
        super().__init__(task_id, filename, model_name)
        self.detect_seal = False


class VLLMProcessor(BaseProcessor):
    """
    Processor for handling OCR tasks with vision-capable LLM models.
    Supports OpenRouter, GLM, Ollama, and other vision models.
    """

    # Model name aliases
    MODEL_ALIASES = {
        'glm-ocr': 'glm',
        'paddle-vl': 'paddle-vl',
        'openrouter': 'openrouter',
        'ollama-glm': 'ollama',
        'ollama': 'ollama',
        'noctrix': 'noctrix',
        'lightonocr': 'lightonocr',
    }

    def __init__(self):
        super().__init__()
        self.models: Dict[str, Any] = {}
        self._init_models()

    def _init_models(self) -> None:
        """Initialize available models based on configuration."""
        # Try to create each model type
        for internal_name, registry_name in self.MODEL_ALIASES.items():
            try:
                model = ModelRegistry.create(registry_name)
                if model:
                    self.models[internal_name] = model
                    logger.info(f"Initialized model: {internal_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize model {internal_name}: {e}")

        # If no models available, use default
        if not self.models:
            self.models['openrouter'] = ModelRegistry.get_default_model()

    def get_model(self, model_name: str) -> Optional[Any]:
        """Get model by name, with alias resolution."""
        resolved_name = self.MODEL_ALIASES.get(model_name, model_name)
        return self.models.get(model_name) or self.models.get(resolved_name)

    def submit_job(
        self,
        filename: str,
        content: bytes,
        model_name: str = 'openrouter',
        detect_seal: bool = False,
        **kwargs
    ) -> Dict[str, str]:
        """
        Submit a new OCR job for processing.

        Args:
            filename: Original filename
            content: File content bytes
            model_name: Name of the model to use
            detect_seal: Whether to detect seals
            **kwargs: Additional arguments

        Returns:
            Dict with job_id
        """
        task_id = str(uuid.uuid4())
        task = VLLMTask(task_id, filename, model_name)
        task.detect_seal = detect_seal
        self.tasks[task_id] = task

        thread = threading.Thread(
            target=self._process_task,
            args=(task_id, content)
        )
        thread.start()

        return {'job_id': task_id}

    def _process_task(self, task_id: str, content: bytes) -> None:
        """Process OCR task with vision model."""
        task = self.tasks[task_id]
        task.status = 'processing'
        task.updated_at = time.time()

        logger.info(
            f"Starting processing task {task_id} for file {task.filename} "
            f"using model {task.model_name}"
        )

        try:
            # Convert file to images
            images = []
            if task.filename.lower().endswith('.pdf'):
                logger.info(f"Converting PDF {task.filename} to images")
                images = convert_from_bytes(content)
            else:
                logger.info(f"Opening image {task.filename}")
                images = [PIL.Image.open(io.BytesIO(content))]

            task.total_pages = len(images)
            if task.total_pages == 0:
                raise ValueError(f"No pages could be extracted from {task.filename}")

            logger.info(f"Total pages to process: {task.total_pages}")

            # Get model
            model = self.get_model(task.model_name)
            if not model:
                raise ValueError(f"Model {task.model_name} not available")

            # Initialize document handler registry
            handlers = DocumentHandlerRegistry.get_all_handlers()

            # Process each page
            pages_results = []
            for i, img in enumerate(images):
                logger.info(f"Processing page {i + 1}/{task.total_pages}")

                # Store image
                img_io = io.BytesIO()
                img.save(img_io, 'PNG')
                img_io.seek(0)
                task.images[i + 1] = img_io.getvalue()

                # Extract text and tables
                result_data = model.extract_text_and_tables(img)

                # Build markdown
                markdown = self._build_markdown(
                    page_num=i + 1,
                    result_data=result_data
                )

                # Detect seals if enabled
                seals = []
                if task.detect_seal:
                    seals = self._detect_seals(img)

                pages_results.append({
                    'page_num': i + 1,
                    'markdown': markdown,
                    'result_data': result_data,
                    'result_json': [],
                    'seals': seals
                })

                task.processed_pages += 1
                task.progress = int((task.processed_pages / task.total_pages) * 100)
                task.updated_at = time.time()

            # Parse document structure
            parsed_data = self._parse_documents(pages_results, handlers)

            task.result = {
                'job_id': task_id,
                'status': 'completed',
                'pages': pages_results,
                'total_pages': task.total_pages,
                'processed_pages': task.processed_pages,
                'parsed_data': parsed_data,
                'detect_seal': task.detect_seal
            }
            task.status = 'completed'
            logger.info(f"Task {task_id} completed successfully")

        except Exception as e:
            logger.exception(f"Error processing task {task_id}: {e}")
            task.status = 'failed'
            task.error = str(e)

        task.updated_at = time.time()

    def _build_markdown(self, page_num: int, result_data: Dict) -> str:
        """Build markdown from extraction result."""
        markdown = f"## Результаты страница {page_num}\n\n"

        if isinstance(result_data, dict):
            if 'text' in result_data and result_data['text']:
                markdown += result_data['text'] + "\n\n"

            if 'tables' in result_data and result_data['tables']:
                markdown += "### Извлечённые таблицы\n\n"
                for table in result_data['tables']:
                    if isinstance(table, list):
                        for row in table:
                            if isinstance(row, list):
                                markdown += "| " + " | ".join([str(c) for c in row]) + " |\n"
                            else:
                                markdown += f"| {row} |\n"
                        markdown += "\n"
        elif isinstance(result_data, list):
            for table in result_data:
                if isinstance(table, list):
                    for row in table:
                        if isinstance(row, list):
                            markdown += "| " + " | ".join([str(c) for c in row]) + " |\n"
                        else:
                            markdown += f"| {row} |\n"
                    markdown += "\n"
        else:
            markdown += str(result_data)

        return markdown

    def _detect_seals(self, image: PIL.Image.Image) -> List[Dict]:
        """Detect seals in image."""
        try:
            detector = get_seal_detector()
            if not detector.is_available():
                return []

            results = detector.detect(image)
            return [
                {
                    'bbox': r.bbox,
                    'confidence': r.confidence,
                    'seal_type': r.seal_type
                }
                for r in results
            ]
        except Exception as e:
            logger.warning(f"Seal detection failed: {e}")
            return []

    def _parse_documents(
        self,
        pages_results: List[Dict],
        handlers: Dict
    ) -> Dict[str, Any]:
        """
        Parse document structure using document handlers.
        """
        # Combine text from all pages
        full_text = ""
        for page in pages_results:
            full_text += page.get('markdown', '') + "\n"

        # Detect document type
        doc_type, confidence = DocumentHandlerRegistry.detect_document_type(full_text)

        if not doc_type:
            return {
                'document_type': None,
                'type_confidence': 0.0,
                'fields': [],
                'tables': []
            }

        # Get handler for detected document type
        handler = DocumentHandlerRegistry.get_handler(doc_type)
        if not handler:
            return {
                'document_type': doc_type,
                'type_confidence': confidence,
                'fields': [],
                'tables': []
            }

        # Parse fields
        regions = self._extract_regions(full_text)
        fields = handler.extract_fields(full_text, regions)

        # Extract tables
        tables = []
        for page in pages_results:
            if 'tables' in page and page['tables']:
                tables.extend(page['tables'])

        return {
            'document_type': doc_type,
            'type_confidence': confidence,
            'fields': fields,
            'tables': tables
        }

    def _extract_regions(self, text: str) -> Dict[str, str]:
        """Extract document regions for field extraction."""
        lines = text.split('\n')
        table_start = len(lines)

        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            if any(
                re.search(p, line_lower)
                for p in [
                    r'^\s*№?\s*(?:наименование|товар|описание|ед\.|кол-во|количество|сумма)',
                    r'^\s*\d+\s+\d+\s+\d+',
                    r'^\s*<table',
                    r'^\s*\|',
                ]
            ):
                table_start = i
                break

        # Find sections
        provider_start = len(lines)
        customer_start = len(lines)
        bank_start = len(lines)

        for i, line in enumerate(lines):
            line_l = line.lower()
            if provider_start == len(lines) and any(kw in line_l for kw in ['поставщик', 'продавец', 'исполнитель']):
                provider_start = i
            if customer_start == len(lines) and any(kw in line_l for kw in ['заказчик', 'покупатель', 'получатель']):
                customer_start = i
            if bank_start == len(lines) and 'банк' in line_l:
                bank_start = i

        sections = sorted([
            ('provider', provider_start),
            ('customer', customer_start),
            ('bank', bank_start),
            ('table', table_start)
        ], key=lambda x: x[1])

        result = {
            'header': '\n'.join(lines[:min(provider_start, customer_start, bank_start, table_start)]),
            'provider': '',
            'customer': '',
            'bank': '',
            'table': '\n'.join(lines[table_start:]),
        }

        for i in range(len(sections) - 1):
            name, start = sections[i]
            next_name, next_start = sections[i+1]
            if start < len(lines):
                result[name] = '\n'.join(lines[start:next_start])

        return result
