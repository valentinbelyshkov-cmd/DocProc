import os
from datetime import timedelta

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff', 'webp', 'zip'}
SECRET_KEY = os.environ.get('SECRET_KEY', 'default-secret-key')
PERMANENT_SESSION_LIFETIME = timedelta(hours=1)
PADDLEOCR_API_URL = os.environ.get('PADDLEOCR_API_URL', 'http://paddleocr-api:8000')
MAX_CONTENT_LENGTH = 64 * 1024 * 1024  # 64 MB limit

# VLLM API Keys
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY', '')
ZHIPUAI_API_KEY = os.environ.get('ZHIPUAI_API_KEY', '')
PADDLEOCR_VL_ENDPOINT = os.environ.get('PADDLEOCR_VL_ENDPOINT', '')

# Ollama local LLM settings
OLLAMA_BASE_URL = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', 'glm-ocr')
# Use the full model name as shown in `ollama list`
NOCTRIX_MODEL = os.environ.get('NOCTRIX_MODEL', 'hf.co/noctrex/LightOnOCR-2-1B-GGUF:Q4_K_M')
