"""
Specialized Ollama model for LightOnOCR-2.
Uses /api/generate endpoint instead of /api/chat for compatibility.
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
    LightOnOCR-2 model via Ollama using /api/generate endpoint.
    Optimized for document OCR tasks.
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

        super().__init__(config)
        self.base_url = base_url or app_config.OLLAMA_BASE_URL
        self.model_name = model_name or app_config.NOCTRIX_MODEL
        self.name = f"lightonocr-{self.model_name}"

        # Load context window size from config
        self.num_ctx = app_config.OCR_MODEL_CONFIG.get('num_ctx', 8192)
        self.stop_sequences = self.config.stop_sequences

    def _image_to_base64(self, image: PIL.Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def generate(
        self,
        prompt: str,
        image: Optional[PIL.Image.Image] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> GenerationResult:
        """
        Generate response using Ollama /api/generate endpoint.
        This endpoint is more compatible with vision models like LightOnOCR.
        """
        # Prepare request payload for /api/generate
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,  # Lower for OCR consistency
                "num_predict": self.config.max_tokens,
                "num_ctx": self.num_ctx,  # CRITICAL: context window size
                "repeat_penalty": self.config.repetition_penalty,  # Higher penalty to prevent repetition
            }
        }

        # Add stop sequences if configured
        if self.stop_sequences:
            payload["stop"] = self.stop_sequences

        # Add image as base64 for vision models
        if image:
            base64_image = self._image_to_base64(image)
            # Use images parameter at root level for /api/generate
            payload["images"] = [base64_image]

        try:
            logger.info(f"LightOnOCR request: {self.base_url}/api/generate (model: {self.model_name})")

            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.config.timeout
            )
            
            # Check for HTTP errors with detailed logging
            if response.status_code >= 400:
                error_detail = ""
                try:
                    error_json = response.json()
                    error_detail = error_json.get('error', '') or str(error_json)
                except:
                    error_detail = response.text[:500] if response.text else "No response body"
                
                logger.error(f"LightOnOCR HTTP error {response.status_code}: {error_detail}")
                
                # Provide helpful error messages based on status code
                if response.status_code == 500:
                    available = self.list_available_models()
                    available_models = [m.get('name', 'unknown') for m in (available or [])]
                    raise ValueError(
                        f"Ollama сервер вернул ошибку 500. Возможные причины: "
                        f"1) Модель '{self.model_name}' не установлена; "
                        f"2) Недостаточно GPU памяти; "
                        f"3) Модель загружается. "
                        f"Доступные модели: {available_models}. "
                        f"Установите модель: ollama pull {self.model_name}"
                    )
                elif response.status_code == 404:
                    raise ValueError(
                        f"Модель '{self.model_name}' не найдена в Ollama. "
                        f"Установите её: ollama pull {self.model_name}"
                    )
                else:
                    raise ValueError(f"Ollama ошибка {response.status_code}: {error_detail}")
            
            response.raise_for_status()

            result = response.json()
            content = result.get('response', '')

            return GenerationResult(
                content=content,
                raw_response=result,
                tokens_used=result.get('eval_count', 0),
                finish_reason=result.get('done_reason', 'stop'),
                model_name=self.model_name
            )

        except requests.exceptions.ConnectionError:
            msg = f"Ollama не подключен по адресу {self.base_url}. Проверьте, запущен ли Ollama (ollama serve) и указан ли корректный OLLAMA_BASE_URL."
            logger.error(msg)
            return GenerationResult(content="", error=msg)

        except requests.exceptions.Timeout:
            return GenerationResult(content="", error="Таймаут запроса к Ollama")

        except requests.exceptions.RequestException as e:
            logger.error(f"LightOnOCR request failed: {e}")
            return GenerationResult(content="", error=f"Ошибка Ollama: {str(e)}")

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
        return """Извлеки текст с этого документа.

ПРАВИЛА:
1. Извлеки ВЕСЬ видимый текст БЕЗ изменений
2. Сохрани структуру: заголовки, параграфы, таблицы
3. Используй ТОЛЬКО допустимые символы: А-Яа-яЁё A-Z a-z 0-9 пробел . , ; : ( ) - / + =

ФОРМАТ ОТВЕТА (ТОЛЬКО JSON):
```json
{"text": "весь извлечённый текст", "tables": [["заголовок1", "заголовок2"], ["ячейка1", "ячейка2"]]}
```

ВОИЗБЕГАЙ повторений! Если текст повторяется - остановись."""

    def list_available_models(self) -> Optional[list]:
        """List available models in Ollama."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            return response.json().get('models', [])
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return None