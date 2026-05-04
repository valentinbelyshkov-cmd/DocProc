"""
Ollama local LLM model implementation.
Supports local models with vision capabilities.
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


class OllamaModel(BaseModel):
    """
    Ollama local LLM wrapper.
    Supports models like GLM-OCR, Llava, etc.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        config: Optional[ModelConfig] = None
    ):
        super().__init__(config or ModelConfig.for_ocr())
        self.base_url = base_url or app_config.OLLAMA_BASE_URL
        self.model_name = model_name or app_config.OLLAMA_MODEL
        self.name = f"ollama-{self.model_name}"

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
        Generate response using Ollama API.
        Supports both text and image inputs for vision models.
        """
        base64_image = None
        if image:
            base64_image = self._image_to_base64(image)

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        user_content = prompt
        if base64_image:
            user_content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ]

        messages.append({"role": "user", "content": user_content})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
        }

        # Apply generation config
        payload["options"] = {
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "num_predict": self.config.max_tokens,
            "stop": self.config.stop_sequences,
        }

        # Repetition penalty (Ollama uses repeat_penalty)
        if self.config.repetition_penalty != 1.0:
            payload["options"]["repeat_penalty"] = self.config.repetition_penalty

        # Frequency/presence penalties
        if self.config.frequency_penalty != 0.0:
            payload["options"]["frequency_penalty"] = self.config.frequency_penalty
        if self.config.presence_penalty != 0.0:
            payload["options"]["presence_penalty"] = self.config.presence_penalty

        try:
            logger.info(f"Ollama request: {self.base_url}/api/chat (model: {self.model_name})")

            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()

            result = response.json()
            content = result.get('message', {}).get('content', '')

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
            logger.error(f"Ollama request failed: {e}")
            return GenerationResult(content="", error=f"Ошибка Ollama: {str(e)}")

        except Exception as e:
            return GenerationResult(content="", error=f"Неизвестная ошибка: {str(e)}")

    def extract_text_and_tables(
        self,
        image: PIL.Image.Image,
        custom_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract text and tables using Ollama model."""
        prompt = custom_prompt or self._get_russian_ocr_prompt()
        result = self.generate(prompt, image=image)

        if not result.success:
            return {
                'text': f"Ошибка: {result.error}",
                'tables': [],
                'raw': None
            }

        content = result.clean_output()

        is_valid, issues = result.validate_content(self.config)
        if not is_valid:
            logger.warning(f"Content validation issues: {issues}")

        return self._parse_response(content)

    def _get_russian_ocr_prompt(self) -> str:
        """Get OCR prompt with Russian language requirements for Ollama."""
        return """Извлеките текст из документа.

ТРЕБОВАНИЯ:
- Используйте ТОЛЬКО русские буквы и арабские цифры
- НЕ добавляйте свой текст
- Таблицы в формате JSON

ФОРМАТ:
{
    "text": "текст документа",
    "tables": [["заголовок", "данные"]]
}"""

    def list_available_models(self) -> Optional[list]:
        """List available models in Ollama."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            return response.json().get('models', [])
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return None


class NoctrixLightOnOCRModel(OllamaModel):
    """
    Noctrix/LightOnOCR-2-1B model via Ollama.
    Specialized OCR model.
    """

    def __init__(self, base_url: Optional[str] = None, config: Optional[ModelConfig] = None):
        super().__init__(
            base_url=base_url,
            model_name=app_config.NOCTRIX_MODEL,
            config=config or ModelConfig.for_ocr()
        )
        self.name = "noctrix-lightonocr"

    def extract_text_and_tables(
        self,
        image: PIL.Image.Image,
        custom_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract text using specialized OCR model."""
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
        """Get optimized prompt for OCR model."""
        return """Perform OCR on this image. Extract all text and tables.
Return as JSON with 'text' and 'tables' fields.
ONLY use Russian (Cyrillic), English letters, and Arabic digits.
"""