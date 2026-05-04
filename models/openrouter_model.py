"""
OpenRouter API model implementation.
Supports multiple LLM providers via OpenRouter gateway.
"""
from typing import Optional, Dict, Any, List
import requests
import base64
import logging
import json
from io import BytesIO
import PIL.Image

from models.base_model import BaseModel, ModelConfig, GenerationResult
import config as app_config

logger = logging.getLogger(__name__)


class OpenRouterModel(BaseModel):
    """
    OpenRouter API model wrapper.
    Supports models from Google, Anthropic, Meta, and other providers.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "google/gemini-flash-1.5",
        config: Optional[ModelConfig] = None
    ):
        super().__init__(config or ModelConfig.for_ocr())
        self.api_key = api_key or app_config.OPENROUTER_API_KEY
        self.model = model
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        self.name = f"openrouter-{model}"

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
        Generate response using OpenRouter API.
        Supports both text-only and image inputs.
        """
        if not self.api_key:
            return GenerationResult(
                content="",
                error="OpenRouter API key is missing"
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://pdf-ocr.local",
            "X-Title": "PDF OCR Converter"
        }

        # Build message content
        content = [{"type": "text", "text": prompt}]

        if image:
            base64_image = self._image_to_base64(image)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"
                }
            })

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        # Build generation parameters
        generation_params = {
            "model": self.model,
            "messages": messages,
        }

        # Apply config with hallucination prevention
        generation_params.update(self._build_generation_params())

        try:
            logger.info(f"OpenRouter request: model={self.model}")
            response = requests.post(
                self.url,
                headers=headers,
                json=generation_params,
                timeout=self.config.timeout
            )
            response.raise_for_status()

            result = response.json()
            content = result['choices'][0]['message']['content']
            finish_reason = result['choices'][0].get('finish_reason', 'stop')

            usage = result.get('usage', {})
            tokens = usage.get('total_tokens', 0)

            return GenerationResult(
                content=content,
                raw_response=result,
                tokens_used=tokens,
                finish_reason=finish_reason,
                model_name=self.model
            )

        except requests.exceptions.Timeout:
            return GenerationResult(content="", error="Таймаут запроса к API")
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenRouter request failed: {e}")
            return GenerationResult(content="", error=f"Ошибка API: {str(e)}")
        except Exception as e:
            return GenerationResult(content="", error=f"Неизвестная ошибка: {str(e)}")

    def _build_generation_params(self) -> Dict[str, Any]:
        """Build generation parameters from config."""
        params = {}

        # Core parameters
        params["max_tokens"] = self.config.max_tokens
        params["temperature"] = self.config.temperature
        params["top_p"] = self.config.top_p
        params["frequency_penalty"] = self.config.frequency_penalty
        params["presence_penalty"] = self.config.presence_penalty

        # Repetition penalty (key anti-hallucination parameter)
        if self.config.repetition_penalty != 1.0:
            params["repetition_penalty"] = self.config.repetition_penalty

        # Stop sequences
        if self.config.stop_sequences:
            params["stop"] = self.config.stop_sequences

        return params

    def extract_text_and_tables(
        self,
        image: PIL.Image.Image,
        custom_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract text and tables from image using OpenRouter.
        Uses optimized OCR prompt with Russian language requirement.
        """
        prompt = custom_prompt or self._get_russian_ocr_prompt()
        result = self.generate(prompt, image=image)

        if not result.success:
            return {
                'text': f"Ошибка: {result.error}",
                'tables': [],
                'raw': None
            }

        content = result.clean_output()

        # Validate output
        is_valid, issues = result.validate_content(self.config)
        if not is_valid:
            logger.warning(f"Content validation issues: {issues}")

        return self._parse_response(content)

    def _get_russian_ocr_prompt(self) -> str:
        """
        Get OCR prompt with Russian language and anti-hallucination instructions.
        """
        return """Выполните OCR (распознавание текста) с изображения документа.

СТРОГИЕ ТРЕБОВАНИЯ:
1. Извлеките ВЕСЬ текст документа БЕЗ ИЗМЕНЕНИЙ
2. Сохраните порядок строк и структуру текста
3. Используйте ТОЛЬКО русские буквы (а-яёА-ЯЁ), английские буквы (a-zA-Z) и арабские цифры (0-9)
4. НЕ ДОБАВЛЯЙТЕ свой текст, комментарии или пояснения
5. НЕ ИЗОБРЕТАЙТЕ данные - если текст нечитаем, напишите [НЕЧИТАЕМО]
6. Таблицы извлекайте построчно

ВЫХОДНОЙ ФОРМАТ (только JSON):
{
    "text": "полный распознанный текст документа",
    "tables": [["заголовок1", "заголовок2"], ["данные1", "данные2"]]
}

Примеры допустимых символов: АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюяABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,;:()[]{}-
Примеры НЕдопустимых символов: ①②③★☆✓✗→←©®™ § «» "" '' ‹› … —–
"""


class OpenRouterClaudeModel(OpenRouterModel):
    """OpenRouter wrapper for Claude models (Anthropic)."""

    def __init__(self, api_key: Optional[str] = None, config: Optional[ModelConfig] = None):
        super().__init__(
            api_key=api_key,
            model="anthropic/claude-3-haiku",
            config=config or ModelConfig.for_ocr()
        )

    def _build_generation_params(self) -> Dict[str, Any]:
        """Claude doesn't use repetition_penalty, use penalty_weight instead."""
        params = super()._build_generation_params()
        params.pop("repetition_penalty", None)
        return params


class OpenRouterGeminiModel(OpenRouterModel):
    """OpenRouter wrapper for Google Gemini models."""

    def __init__(self, api_key: Optional[str] = None, config: Optional[ModelConfig] = None):
        super().__init__(
            api_key=api_key,
            model="google/gemini-flash-1.5",
            config=config or ModelConfig.for_ocr()
        )