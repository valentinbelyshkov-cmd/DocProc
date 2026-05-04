from abc import ABC, abstractmethod
import os
import requests
import json
import base64
from io import BytesIO
import PIL.Image
import config

class BaseVLLMModel(ABC):
    @abstractmethod
    def extract_tables(self, image: PIL.Image.Image) -> list:
        """
        Extracts tables from the given image.
        Returns a list of tables, where each table is a list of rows (list of strings).
        """
        pass

class OpenRouterModel(BaseVLLMModel):
    def __init__(self, api_key=None, model="google/gemini-flash-1.5"):
        self.api_key = api_key or config.OPENROUTER_API_KEY
        self.model = model
        self.url = "https://openrouter.ai/api/v1/chat/completions"

    def _image_to_base64(self, image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def extract_tables(self, image: PIL.Image.Image) -> list:
        if not self.api_key:
            raise ValueError("OpenRouter API key is missing")

        base64_image = self._image_to_base64(image)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        prompt = "Extract all tables from this image and return them as a JSON list of lists (rows and cells). Only return the JSON."
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
        }
        
        response = requests.post(self.url, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        content = result['choices'][0]['message']['content']
        
        # Try to parse JSON from content
        try:
            # Clean up potential markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            return json.loads(content)
        except Exception as e:
            print(f"Error parsing VLLM response: {e}")
            return [[["Error parsing response", str(e)]]]

class GLMOCRModel(BaseVLLMModel):
    def __init__(self, api_key=None):
        self.api_key = api_key or config.ZHIPUAI_API_KEY
        # GLM-OCR is often accessed via ZhipuAI API
        self.url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

    def _image_to_base64(self, image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def extract_tables(self, image: PIL.Image.Image) -> list:
        if not self.api_key:
            raise ValueError("ZhipuAI (GLM) API key is missing")

        base64_image = self._image_to_base64(image)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "glm-4v", # Or a specific GLM-OCR model if available
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Identify and extract all tables from this image. Output in JSON format as a list of tables."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": base64_image
                            }
                        }
                    ]
                }
            ]
        }
        
        response = requests.post(self.url, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        content = result['choices'][0]['message']['content']
        
        try:
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            return json.loads(content)
        except:
            return [[["Failed to parse GLM response"]]]

class PaddleOCRVLModel(BaseVLLMModel):
    def __init__(self, endpoint_url=None):
        self.endpoint_url = endpoint_url or config.PADDLEOCR_VL_ENDPOINT

    def extract_tables(self, image: PIL.Image.Image) -> list:
        # Assuming PaddleOCR-VL-1.5 is hosted as a service
        if not self.endpoint_url:
            return [[["PaddleOCR-VL-1.5 endpoint not configured"]]]
            
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        files = {'image': buffered.getvalue()}
        
        try:
            response = requests.post(self.endpoint_url, files=files)
            response.raise_for_status()
            return response.json().get('tables', [])
        except Exception as e:
            return [[["Error calling PaddleOCR-VL", str(e)]]]


class OllamaModel(BaseVLLMModel):
    """Ollama local LLM with vision support (GLM-OCR, Noctrix LightOnOCR-2-1B, etc.)"""
    
    def __init__(self, model_name="glm-ocr", base_url=None):
        self.model_name = model_name or config.OLLAMA_MODEL
        self.base_url = base_url or config.OLLAMA_BASE_URL

    def _image_to_base64(self, image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def extract_tables(self, image: PIL.Image.Image) -> list:
        import base64
        
        base64_image = self._image_to_base64(image)
        
        # Ollama API format
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": "Extract all tables from this image and return them as a JSON list of lists (rows and cells). Only return valid JSON, no markdown.",
                    "images": [base64_image]
                }
            ],
            "stream": False
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            content = result.get('message', {}).get('content', '')
            
            # Parse JSON from response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            return json.loads(content)
        except requests.exceptions.ConnectionError:
            return [[["Ollama не подключен. Запустите: ollama serve"]]]
        except requests.exceptions.Timeout:
            return [[["Таймаут запроса к Ollama"]]]
        except Exception as e:
            return [[["Error calling Ollama", str(e)]]]


class NoctrixLightOnOCRModel(BaseVLLMModel):
    """Noctrix/LightOnOCR-2-1B model via Ollama"""
    
    def __init__(self, base_url=None):
        self.base_url = base_url or config.OLLAMA_BASE_URL
        self.model_name = config.NOCTRIX_MODEL

    def _image_to_base64(self, image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def extract_tables(self, image: PIL.Image.Image) -> list:
        base64_image = self._image_to_base64(image)
        
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": "Perform OCR on this image. Extract all text and tables. Return as JSON format with 'text' and 'tables' fields. Only return valid JSON.",
                    "images": [base64_image]
                }
            ],
            "stream": False
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=180
            )
            response.raise_for_status()
            
            result = response.json()
            content = result.get('message', {}).get('content', '')
            
            # Parse JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            return json.loads(content)
        except Exception as e:
            return [[["Error calling Noctrix LightOnOCR", str(e)]]]
