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

        # Load context window size from config
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
        Generate response using Ollama API.
        Supports both text and image inputs for vision models.
        """
        # Try to resolve model name if not already done
        self._resolve_model_name()

        base64_image = None
        if image:
            base64_image = self._image_to_base64(image)

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        user_message = {"role": "user"}
        if prompt:
            user_message["content"] = prompt
            
        if base64_image:
            user_message["images"] = [base64_image]

        messages.append(user_message)

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
        }

        # Apply generation config
        payload["options"] = {
            "temperature": 0.2 if "LightOnOCR" in self.model_name or "lighton" in self.model_name.lower() else self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "num_predict": 4096 if "LightOnOCR" in self.model_name or "lighton" in self.model_name.lower() else self.config.max_tokens,
            "num_ctx": 16384 if "LightOnOCR" in self.model_name or "lighton" in self.model_name.lower() or "glm" in self.model_name.lower() else self.num_ctx,
            "stop": self.config.stop_sequences,
        }

        # Repetition penalty (Ollama uses repeat_penalty) - set high to prevent loops
        payload["options"]["repeat_penalty"] = max(self.config.repetition_penalty, 1.4)

        # Frequency/presence penalties
        if self.config.frequency_penalty != 0.0:
            payload["options"]["frequency_penalty"] = self.config.frequency_penalty
        if self.config.presence_penalty != 0.0:
            payload["options"]["presence_penalty"] = self.config.presence_penalty

        try:
            logger.info(f"Ollama request: {self.base_url}/api/chat (model: {self.model_name})")
            
            # Log payload for debugging (with truncated images to keep logs clean)
            log_payload = json.loads(json.dumps(payload))
            for msg in log_payload.get("messages", []):
                if "images" in msg and msg["images"]:
                    img_data = msg["images"][0]
                    msg["images"] = [f"{img_data[:50]}...[truncated {len(img_data)} chars]"]
            
            logger.info(f"Ollama payload: {json.dumps(log_payload, ensure_ascii=False, indent=2)}")

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
                
                logger.error(f"Ollama HTTP error {response.status_code}: {error_detail}")
                
                # Provide helpful error messages based on status code
                if response.status_code == 500:
                    # 500 errors often mean model not loaded or server issues
                    # Check for OOM in error detail
                    is_oom = any(kw in error_detail.lower() for kw in ['out of memory', 'oom', 'gpu', 'memory'])
                    
                    # Try to list available models to help diagnose
                    available = self.list_available_models()
                    available_models = [m.get('name', 'unknown') for m in (available or [])]
                    
                    if is_oom:
                        raise ValueError(
                            f"Ollama сервер вернул ошибку 500 (OOM). Недостаточно GPU памяти для модели '{self.model_name}' с контекстом {self.num_ctx}. "
                            f"Попробуйте уменьшить OLLAMA_NUM_CTX в настройках."
                        )
                    
                    raise ValueError(
                        f"Ollama сервер вернул ошибку 500. Возможные причины: "
                        f"1) Модель '{self.model_name}' не установлена или повреждена; "
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

        except ValueError as e:
            return GenerationResult(content="", error=str(e))

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
        return """Извлеки текст с этого документа.

ПРАВИЛА:
1. Извлеки ВЕСЬ видимый текст БЕЗ изменений
2. Сохрани структуру: заголовки, параграфы, таблицы
3. Используй русский и английский языки, цифры и основные знаки пунктуации (включая подчеркивание _)

ФОРМАТ ОТВЕТА (ТОЛЬКО JSON):
```json
{"text": "весь извлечённый текст", "tables": [["заголовок1", "заголовок2"], ["ячейка1", "ячейка2"]]}
```

НЕ пиши ничего, кроме JSON-блока. ВОИЗБЕГАЙ повторений! Если текст повторяется - остановись."""

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
        return """Извлеки текст с этого документа.

ПРАВИЛА:
1. Извлеки ВЕСЬ видимый текст БЕЗ изменений
2. Сохрани структуру: заголовки, параграфы, таблицы
3. Используй русский и английский языки, цифры и основные знаки пунктуации (включая подчеркивание _)

ФОРМАТ ОТВЕТА (ТОЛЬКО JSON):
```json
{"text": "весь извлечённый текст", "tables": [["заголовок1", "заголовок2"], ["ячейка1", "ячейка2"]]}
```

НЕ пиши ничего, кроме JSON-блока. ВОИЗБЕГАЙ повторений! Если текст повторяется - остановись."""