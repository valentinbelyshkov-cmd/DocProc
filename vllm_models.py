from abc import ABC, abstractmethod
import os
import requests
import json
import base64
from io import BytesIO
import PIL.Image
import config

import logging

logger = logging.getLogger(__name__)

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
            logger.error("OpenRouter API key is missing")
            raise ValueError("OpenRouter API key is missing")

        base64_image = self._image_to_base64(image)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        prompt = "Extract all text and tables from this image. Output tables in JSON format as a list of lists. Return everything in a JSON object with 'text' and 'tables' fields."
        
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
        
        try:
            logger.info(f"Sending request to OpenRouter API (model: {self.model})")
            response = requests.post(self.url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            content = result['choices'][0]['message']['content']
            logger.info("Received response from OpenRouter")
            
            # Try to parse JSON from content
            try:
                json_content = content
                if "```json" in content:
                    json_content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    json_content = content.split("```")[1].split("```")[0].strip()
                
                parsed = json.loads(json_content)
                if isinstance(parsed, dict):
                    return parsed
                return {"text": content, "tables": parsed if isinstance(parsed, list) else []}
            except Exception as e:
                logger.warning(f"Failed to parse JSON from OpenRouter response: {e}")
                return {"text": content, "tables": []}
        except Exception as e:
            logger.error(f"Error calling OpenRouter API: {e}")
            raise

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
            logger.error("ZhipuAI (GLM) API key is missing")
            raise ValueError("ZhipuAI (GLM) API key is missing")

        base64_image = self._image_to_base64(image)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Correctly format for GLM-4V: use data:image prefix if needed or just base64 depending on API version
        # For ZhipuAI API v4, it usually expects the data URL format for base64 images
        payload = {
            "model": "glm-4v", 
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract all text and tables from this image. Output tables in JSON format as a list of lists. Return both text and tables."},
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
        
        try:
            logger.info(f"Sending request to ZhipuAI GLM API: {self.url}")
            response = requests.post(self.url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            content = result['choices'][0]['message']['content']
            logger.info("Successfully received response from ZhipuAI")
            
            try:
                # Try to find JSON in the content
                json_content = content
                if "```json" in content:
                    json_content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    json_content = content.split("```")[1].split("```")[0].strip()
                
                try:
                    parsed = json.loads(json_content)
                    if isinstance(parsed, dict):
                        return parsed
                    return {"text": content, "tables": parsed if isinstance(parsed, list) else []}
                except json.JSONDecodeError:
                    return {"text": content, "tables": []}
            except Exception as e:
                logger.warning(f"Failed to parse JSON from GLM response: {e}")
                return {"text": content, "tables": []}
        except Exception as e:
            logger.error(f"Error calling ZhipuAI GLM API: {e}")
            raise

class PaddleOCRVLModel(BaseVLLMModel):
    def __init__(self, endpoint_url=None):
        self.endpoint_url = endpoint_url or config.PADDLEOCR_VL_ENDPOINT

    def extract_tables(self, image: PIL.Image.Image) -> list:
        # Assuming PaddleOCR-VL-1.5 is hosted as a service
        if not self.endpoint_url:
            logger.error("PaddleOCR-VL-1.5 endpoint not configured")
            return {"text": "PaddleOCR-VL-1.5 endpoint not configured", "tables": []}
            
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        files = {'image': ('image.png', buffered.getvalue(), 'image/png')}
        
        try:
            logger.info(f"Sending request to PaddleOCR-VL at {self.endpoint_url}")
            response = requests.post(self.endpoint_url, files=files, timeout=60)
            response.raise_for_status()
            data = response.json()
            logger.info("Received response from PaddleOCR-VL")
            return {
                "text": data.get('text', ''),
                "tables": data.get('tables', [])
            }
        except Exception as e:
            logger.error(f"Error calling PaddleOCR-VL: {e}")
            return {"text": f"Error calling PaddleOCR-VL: {str(e)}", "tables": []}


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
                    "content": "Extract all text and tables from this image. Output tables in JSON format as a list of lists. Return everything as a JSON object with 'text' and 'tables' fields.",
                    "images": [base64_image]
                }
            ],
            "stream": False
        }
        
        try:
            logger.info(f"Sending request to Ollama API: {self.base_url}/api/chat (model: {self.model_name})")
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            content = result.get('message', {}).get('content', '')
            logger.info(f"Received response from Ollama: {len(content)} chars")
            
            # Parse JSON from response
            try:
                json_content = content
                if "```json" in content:
                    json_content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    json_content = content.split("```")[1].split("```")[0].strip()
                
                parsed = json.loads(json_content)
                if isinstance(parsed, dict):
                    return parsed
                return {"text": content, "tables": parsed if isinstance(parsed, list) else []}
            except Exception as e:
                logger.warning(f"Could not parse Ollama response as JSON: {e}")
                return {"text": content, "tables": []}
                
        except requests.exceptions.ConnectionError:
            msg = f"Ollama не подключен по адресу {self.base_url}. Запустите: ollama serve"
            logger.error(msg)
            return {"text": msg, "tables": []}
        except requests.exceptions.Timeout:
            msg = "Таймаут запроса к Ollama"
            logger.error(msg)
            return {"text": msg, "tables": []}
        except Exception as e:
            msg = f"Error calling Ollama: {str(e)}"
            logger.error(msg)
            return {"text": msg, "tables": []}


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
            logger.info(f"Sending request to Noctrix LightOnOCR (Ollama) at {self.base_url}")
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=180
            )
            response.raise_for_status()
            
            result = response.json()
            content = result.get('message', {}).get('content', '')
            logger.info("Received response from Noctrix LightOnOCR")
            
            # Parse JSON
            try:
                json_content = content
                if "```json" in content:
                    json_content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    json_content = content.split("```")[1].split("```")[0].strip()
                
                parsed = json.loads(json_content)
                if isinstance(parsed, dict):
                    return parsed
                return {"text": content, "tables": parsed if isinstance(parsed, list) else []}
            except Exception as e:
                logger.warning(f"Failed to parse Noctrix response as JSON: {e}")
                return {"text": content, "tables": []}
        except Exception as e:
            logger.error(f"Error calling Noctrix LightOnOCR: {e}")
            return {"text": f"Error calling Noctrix LightOnOCR: {str(e)}", "tables": []}
