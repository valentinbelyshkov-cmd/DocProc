import requests
import logging
from requests.exceptions import ConnectionError, Timeout, RequestException
from config import PADDLEOCR_API_URL

logger = logging.getLogger(__name__)

class PaddleOCRClient:
    def __init__(self, base_url=PADDLEOCR_API_URL):
        self.base_url = base_url

    def submit_job(self, filename, content, content_type, detect_seal=False):
        files = {'file': (filename, content, content_type)}
        data = {'detect_seal': 'true' if detect_seal else 'false'}
        response = requests.post(f"{self.base_url}/ocr", files=files, data=data, timeout=30)
        response.raise_for_status()
        return response.json()

    def is_available(self):
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                # Check if model is actually loaded
                return data.get('status') == 'ok' and data.get('PaddleOCR', False)
            return False
        except (ConnectionError, Timeout, RequestException):
            return False

    def get_model_status(self):
        """Get detailed model status from health endpoint"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                return response.json()
            return None
        except (ConnectionError, Timeout, RequestException):
            return None

    def get_status(self, job_id):
        response = requests.get(f"{self.base_url}/ocr/{job_id}")
        response.raise_for_status()
        return response.json()

    def get_result(self, job_id):
        response = requests.get(f"{self.base_url}/ocr/{job_id}/result")
        response.raise_for_status()
        return response.json()

    def get_image(self, job_id, page_num):
        return requests.get(f"{self.base_url}/ocr/{job_id}/image/{page_num}", stream=True)

    def list_seals(self, job_id):
        response = requests.get(f"{self.base_url}/ocr/{job_id}/seals")
        response.raise_for_status()
        return response.json()

    def get_seal(self, job_id, filename):
        return requests.get(f"{self.base_url}/ocr/{job_id}/seals/{filename}", stream=True)
