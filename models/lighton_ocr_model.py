"""
Specialized Ollama model for LightOnOCR-2.
Uses /api/chat endpoint with specific parameters for document OCR.
"""
from typing import Optional, Dict, Any
import requests
import base64
import logging
import json
from io import BytesIO
import PIL.Image

from models.base_model import BaseModel, ModelConfig, GenerationResult
import config as app_config

logger = logging.getLogger(__name__)


class LightOnOCRModel(BaseModel):
    """
    LightOnOCR-2 model via Ollama using /api/chat endpoint.
    Optimized for document OCR tasks with large context.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        config: Optional[ModelConfig] = None
    ):
        # Use default LightOnOCR config if not provided
        if config is None:
            config = ModelConfig.for_ocr()
            # User requested 0.2 temperature and 4096 tokens
            config.temperature = 0.2
            config.max_tokens = 4096

        super().__init__(config)
        self.base_url = base_url or app_config.OLLAMA_BASE_URL
        self.model_name = model_name or app_config.NOCTRIX_MODEL
        self.name = f"lightonocr-{self.model_name}"

        # Load context window size from config (now 16384 by default)
        self.num_ctx = app_config.OCR_MODEL_CONFIG.get('num_ctx', 16384)

    def _image_to_base64(self, image: PIL.Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def _resolve_model_name(self):
        """Try to resolve model name to an available one in Ollama."""
        if getattr(self, "_model_name_resolved", False):
            return

        try:
            available = self.list_available_models()
            if not available:
                return

            available_names = [m.get('name') for m in available if m.get('name')]
            
            # If current model name is already in available models, do nothing
            if self.model_name in available_names:
                self._model_name_resolved = True
                return

            # Try adding :latest if no tag
            if ":" not in self.model_name:
                if f"{self.model_name}:latest" in available_names:
                    logger.info(f"Resolved model name {self.model_name} to {self.model_name}:latest")
                    self.model_name = f"{self.model_name}:latest"
                    self._model_name_resolved = True
                    return

            # Try matching without tag
            for name in available_names:
                if ":" in name and name.split(":")[0] == self.model_name:
                    logger.info(f"Resolved model name {self.model_name} to {name}")
                    self.model_name = name
                    self._model_name_resolved = True
                    return
            
            # Mark as resolved even if no match found, to avoid repeated checks
            self._model_name_resolved = True
        except Exception as e:
            logger.warning(f"Failed to resolve model name: {e}")

    def generate(
        self,
        prompt: str,
        image: Optional[PIL.Image.Image] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> GenerationResult:
        """
        Generate response using Ollama /api/chat endpoint.
        Follows specific rules for maternion/LightOnOCR-2.
        """
        # Try to resolve model name
        self._resolve_model_name()

        # Prepare request payload according to user specification
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "images": [self._image_to_base64(image)] if image else []
                }
            ],
            "stream": False,
            "options": {
                "num_ctx": 16384,      # glm-ocr / lightonocr require large context
                "temperature": 0.0,    # minimal hallucinations
                "num_predict": 4096,   # enough for large table
            }
        }

        try:
            logger.info(f"LightOnOCR request: {self.base_url}/api/chat (model: {self.model_name})")
            
            # Log payload for debugging (with truncated images to keep logs clean)
            log_payload = json.loads(json.dumps(payload))
            for msg in log_payload.get("messages", []):
                if "images" in msg and msg["images"]:
                    img_data = msg["images"][0]
                    msg["images"] = [f"{img_data[:50]}...[truncated {len(img_data)} chars]"]
            
            logger.info(f"LightOnOCR payload: {json.dumps(log_payload, ensure_ascii=False, indent=2)}")

            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.config.timeout
            )
            
            # Check for HTTP errors with detailed logging
            if response.status_code >= 400:
                error_detail = ""
                try:
                    error_json = response.json()
                    error_detail = error_json.get('error', {}).get('message', '') or str(error_json)
                except:
                    error_detail = response.text[:500] if response.text else "No response body"
                
                logger.error(f"LightOnOCR HTTP error {response.status_code}: {error_detail}")
                
                # Provide helpful error messages
                if response.status_code == 500:
                    is_oom = any(kw in error_detail.lower() for kw in ['out of memory', 'oom', 'gpu', 'memory'])
                    if is_oom:
                        raise ValueError(
                            f"Ollama сервер вернул ошибку 500 (OOM). Недостаточно GPU памяти для модели '{self.model_name}' с контекстом 16384."
                        )
                    raise ValueError(f"Ollama сервер вернул ошибку 500: {error_detail}")
                elif response.status_code == 404:
                    raise ValueError(f"Модель '{self.model_name}' не найдена в Ollama.")
                else:
                    raise ValueError(f"Ollama ошибка {response.status_code}: {error_detail}")
            
            response.raise_for_status()

            result = response.json()
            logger.info(f"LightOnOCR raw JSON response: {json.dumps(result, ensure_ascii=False)}")
            content = result.get('message', {}).get('content', '')

            return GenerationResult(
                content=content,
                raw_response=result,
                tokens_used=result.get('eval_count', 0),
                finish_reason=result.get('done_reason', 'completed'),
                model_name=self.model_name
            )

        except requests.exceptions.ConnectionError:
            msg = f"Ollama не подключен по адресу {self.base_url}."
            logger.error(msg)
            return GenerationResult(content="", error=msg)

        except requests.exceptions.Timeout:
            return GenerationResult(content="", error="Таймаут запроса к Ollama")

        except requests.exceptions.RequestException as e:
            logger.error(f"LightOnOCR request failed: {e}")
            return GenerationResult(content="", error=f"Ошибка Ollama: {str(e)}")

        except ValueError as e:
            return GenerationResult(content="", error=str(e))

        except Exception as e:
            return GenerationResult(content="", error=f"Неизвестная ошибка: {str(e)}")

    def extract_text_and_tables(
        self,
        image: PIL.Image.Image,
        custom_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract text and tables using LightOnOCR model."""
        prompt = custom_prompt or self._get_ocr_prompt()
        result = self.generate(prompt, image=image)

        if not result.success:
            return {
                'text': f"Ошибка: {result.error}",
                'tables': [],
                'raw': None
            }

        content = result.clean_output()
        return self._parse_response(content)

    def _get_ocr_prompt(self) -> str:
        """Get optimized prompt for LightOnOCR model."""
        return ""


    def list_available_models(self) -> Optional[list]:
        """List available models in Ollama."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            return response.json().get('models', [])
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return None
