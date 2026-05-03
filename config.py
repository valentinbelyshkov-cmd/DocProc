import os
from datetime import timedelta

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff', 'webp', 'zip'}
SECRET_KEY = os.environ.get('SECRET_KEY', 'default-secret-key')
PERMANENT_SESSION_LIFETIME = timedelta(hours=1)
PADDLEOCR_API_URL = os.environ.get('PADDLEOCR_API_URL', 'http://paddleocr-api:8000')
MAX_CONTENT_LENGTH = 64 * 1024 * 1024  # 64 MB limit
