"""
Configuration file for PDF OCR Converter.
"""
import os
from datetime import timedelta

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'output')
DEBUG_IMAGES_FOLDER = os.path.join(BASE_DIR, 'debug_images')

# File handling
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff', 'webp', 'zip'}
MAX_CONTENT_LENGTH = 64 * 1024 * 1024  # 64 MB limit

# Session settings
SECRET_KEY = os.environ.get('SECRET_KEY', 'default-secret-key')
PERMANENT_SESSION_LIFETIME = timedelta(hours=1)

# VLLM API Keys
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY', '')
ZHIPUAI_API_KEY = os.environ.get('ZHIPUAI_API_KEY', '')

# Ollama local LLM settings
OLLAMA_BASE_URL = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', 'glm-ocr:latest')
NOCTRIX_MODEL = os.environ.get('NOCTRIX_MODEL', 'maternion/LightOnOCR-2:latest')

# Seal detection settings
SEAL_DETECTOR_TYPE = os.environ.get('SEAL_DETECTOR_TYPE', 'yolo')
SEAL_MODEL_PATH = os.environ.get('SEAL_MODEL_PATH', '')
SEAL_CONFIDENCE_THRESHOLD = float(os.environ.get('SEAL_CONFIDENCE_THRESHOLD', '0.5'))

# OCR Model configuration
OCR_MODEL_CONFIG = {
    # Default model
    'default_model': os.environ.get('DEFAULT_OCR_MODEL', 'openrouter'),

    # Generation parameters (anti-hallucination)
    'max_tokens': int(os.environ.get('OCR_MAX_TOKENS', '4096')),
    'temperature': float(os.environ.get('OCR_TEMPERATURE', '0.05')),

    # Context window size for Ollama (CRITICAL for large documents)
    'num_ctx': int(os.environ.get('OLLAMA_NUM_CTX', '16384')),

    # Timeout settings
    'request_timeout': int(os.environ.get('OCR_REQUEST_TIMEOUT', '300')),
    'max_retries': int(os.environ.get('OCR_MAX_RETRIES', '3')),
}

# Document table extraction settings
TABLE_EXTRACTION = {
    'enabled': os.environ.get('TABLE_EXTRACTION_ENABLED', 'true').lower() == 'true',
    'min_rows': int(os.environ.get('TABLE_MIN_ROWS', '2')),
    'min_columns': int(os.environ.get('TABLE_MIN_COLUMNS', '2')),
}

# Task cleanup settings
TASK_CLEANUP = {
    'max_age_seconds': int(os.environ.get('TASK_MAX_AGE_SECONDS', '3600')),
    'cleanup_interval': int(os.environ.get('TASK_CLEANUP_INTERVAL', '300')),
}