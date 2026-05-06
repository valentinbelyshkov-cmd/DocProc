"""
ZhipuAI (GLM-4V) model implementation.
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


class GLMOCRModel(BaseModel):
    """
    ZhipuAI GLM-4V model wrapper.
    Chinese LLM with vision capabilities.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[ModelConfig] = None
    ):
        super().__init__(config or ModelConfig.for_ocr())
        self.api_key = api_key or app_config.ZHIPUAI_API_KEY
        self.url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        self.model = "glm-4v"
        self.name = "glm-4v"

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
        Generate response using ZhipuAI GLM API.
        """
        if not self.api_key:
            return GenerationResult(
                content="",
                error="ZhipuAI API key is missing"
            )

        # Default system prompt if none provided to ensure JSON output
        if not system_prompt:
            system_prompt = (
                "You are a professional OCR system. Your task is to extract all text "
                "from the image and return it strictly in JSON format. "
                "Do not include any explanations or markdown outside the JSON block."
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Build content
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

        payload = {
            "model": self.model,
            "messages": messages,
        }

        # Apply config
        if self.config.temperature:
            payload["temperature"] = self.config.temperature
        if self.config.max_tokens:
            payload["max_tokens"] = self.config.max_tokens
        if self.config.top_p:
            payload["top_p"] = self.config.top_p
        if self.config.stop_sequences:
            payload["stop"] = self.config.stop_sequences

        try:
            logger.info(f"GLM-4V request: {self.url}")
            response = requests.post(
                self.url,
                headers=headers,
                json=payload,
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
            return GenerationResult(content="", error="Таймаут запроса к GLM API")
        except requests.exceptions.RequestException as e:
            logger.error(f"GLM-4V request failed: {e}")
            return GenerationResult(content="", error=f"Ошибка API: {str(e)}")
        except Exception as e:
            return GenerationResult(content="", error=f"Неизвестная ошибка: {str(e)}")

    def extract_text_and_tables(
        self,
        image: PIL.Image.Image,
        custom_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract text and tables using GLM-4V."""
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
        """Get OCR prompt with Russian language requirements."""
        return """Извлеки текст с этого документа и представь его СТРОГО в формате JSON.

ПРАВИЛА:
1. Извлеки ВЕСЬ видимый текст БЕЗ изменений
2. Сохрани структуру: заголовки, параграфы, таблицы
3. Используй ТОЛЬКО допустимые символы: А-Яа-яЁё A-Z a-z 0-9 пробел . , ; : ( ) - / + =
4. Текст должен быть в поле "text"
5. Все таблицы должны быть в поле "tables" как список списков (матрица)

ФОРМАТ ОТВЕТА (ТОЛЬКО ЧИСТЫЙ JSON):
{
  "text": "весь извлечённый текст",
  "tables": [
    ["Колонка 1", "Колонка 2"],
    ["Данные 1", "Данные 2"]
  ]
}

НЕ пиши ничего кроме JSON! ВОИЗБЕГАЙ повторений! Если текст повторяется - остановись."""
