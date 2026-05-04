"""
Base classes for AI models used for OCR and document processing.
Provides configuration and result types for model interactions.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import logging
import re
import PIL.Image

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """
    Configuration for AI model generation parameters.
    Includes settings to prevent hallucinations and control output quality.
    """
    # Token limits
    max_tokens: int = 1024
    min_tokens: int = 16

    # Temperature controls randomness (lower = more deterministic)
    temperature: float = 0.1

    # Repetition penalty prevents infinite loops
    repetition_penalty: float = 1.15

    # Stop sequences for early termination
    stop_sequences: List[str] = field(default_factory=lambda: ["```", "END", "###"])

    # Top-p sampling for output diversity
    top_p: float = 0.95

    # Top-k sampling
    top_k: int = 50

    # Enable early stopping
    early_stopping: bool = True

    # Number of output beams (for beam search)
    num_beams: int = 1

    # Frequency penalty for repeated tokens
    frequency_penalty: float = 0.0

    # Presence penalty for topic diversity
    presence_penalty: float = 0.0

    # Timeout for API calls
    timeout: int = 60

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0

    # Language restrictions (for hallucination prevention)
    allowed_chars_pattern: str = r'^[а-яА-ЯёЁa-zA-Z0-9\s.,:;()\-–—+=/§«»""'']+$'
    allowed_languages: List[str] = field(default_factory=lambda: ['ru', 'en'])

    # Table extraction settings
    table_min_rows: int = 2
    table_min_columns: int = 2
    table_extraction_enabled: bool = True

    @classmethod
    def for_ocr(cls) -> 'ModelConfig':
        """Optimized config for OCR tasks."""
        return cls(
            max_tokens=1024,
            temperature=0.1,
            repetition_penalty=1.15,
            early_stopping=True,
            top_p=0.95,
        )

    @classmethod
    def for_document_parsing(cls) -> 'ModelConfig':
        """Optimized config for document parsing."""
        return cls(
            max_tokens=512,
            temperature=0.05,
            repetition_penalty=1.2,
            early_stopping=True,
            top_p=0.9,
        )

    @classmethod
    def for_table_extraction(cls) -> 'ModelConfig':
        """Optimized config for table extraction."""
        return cls(
            max_tokens=2048,
            temperature=0.1,
            repetition_penalty=1.1,
            early_stopping=True,
            table_extraction_enabled=True,
        )


@dataclass
class GenerationResult:
    """Result from model generation."""
    content: str
    raw_response: Any = None
    tokens_used: int = 0
    finish_reason: str = "stop"
    model_name: Optional[str] = None
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """Check if generation was successful."""
        return self.error is None and bool(self.content)

    def validate_content(self, config: ModelConfig) -> Tuple[bool, List[str]]:
        """
        Validate that content matches expected patterns.
        Returns (is_valid, list_of_issues).
        """
        issues = []

        if not self.content:
            issues.append("Пустой ответ от модели")
            return False, issues

        # Check for character set
        if config.allowed_chars_pattern:
            if not re.match(config.allowed_chars_pattern, self.content):
                issues.append("Ответ содержит недопустимые символы")

        # Check for minimum content
        if len(self.content.strip()) < 5:
            issues.append("Слишком короткий ответ")

        # Check for repetition (potential hallucination)
        words = self.content.lower().split()
        if len(words) >= 10:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                issues.append("Высокий уровень повторений - возможная галлюцинация")

        # Check for common hallucination patterns
        hallucination_patterns = [
            r'^я\s+не\s+могу',
            r'^к\s+сожалению',
            r'^,\s+но\s+это',
            r'^однако\s+я',
        ]
        for pattern in hallucination_patterns:
            if re.search(pattern, self.content.lower()):
                issues.append("Обнаружен паттерн отказа/неуверенности модели")

        return len(issues) == 0, issues

    def clean_output(self, remove_json_wrapper: bool = True) -> str:
        """Clean model output by removing markdown code blocks."""
        content = self.content

        if remove_json_wrapper:
            # Remove markdown code blocks
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]

            if content.endswith("```"):
                content = content[:-3]

            content = content.strip()

        return content


class BaseModel(ABC):
    """
    Abstract base class for AI models used in OCR/document processing.
    All model implementations should inherit from this class.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig.for_ocr()
        self.name: str = "base_model"

    @abstractmethod
    def generate(self, prompt: str, image: Optional[PIL.Image.Image] = None, **kwargs) -> GenerationResult:
        """
        Generate response from model.
        Must be implemented by subclasses.
        """
        pass

    def extract_text_and_tables(self, image: PIL.Image.Image, custom_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract text and tables from image.
        Default implementation uses generate() method.
        """
        prompt = custom_prompt or self._get_default_ocr_prompt()

        result = self.generate(prompt, image=image)

        if not result.success:
            return {
                'text': f"Ошибка: {result.error}",
                'tables': [],
                'raw': None
            }

        # Try to parse JSON from response
        content = result.clean_output()
        return self._parse_response(content)

    def _get_default_ocr_prompt(self) -> str:
        """Get default OCR prompt."""
        return """Извлеките текст из изображения.

Требования:
1. Извлеките ВЕСЬ текст без изменений
2. Сохраните структуру текста
3. Используйте ТОЛЬКО русские буквы и арабские цифры
4. Таблицы представьте в виде списка списков

Формат ответа:
{
    "text": "извлечённый текст",
    "tables": [["строка1", "столбец1", "столбец2"], ["строка2", "данные", "данные"]]
}"""

    def _parse_response(self, content: str) -> Dict[str, Any]:
        """Parse JSON response from model."""
        import json

        # Direct try
        try:
            parsed = json.loads(content)
            return {
                'text': parsed.get('text', content),
                'tables': parsed.get('tables', []),
                'raw': parsed
            }
        except json.JSONDecodeError:
            # Try to extract JSON using regex
            json_match = re.search(r'(\{.*\})', content, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(1))
                    return {
                        'text': parsed.get('text', content),
                        'tables': parsed.get('tables', []),
                        'raw': parsed
                    }
                except json.JSONDecodeError:
                    pass

            logger.warning(f"Не удалось распарсить JSON из ответа модели. Content: {content[:100]}...")
            return {
                'text': content,
                'tables': [],
                'raw': None
            }

    def validate_generation_config(self) -> List[str]:
        """
        Validate the generation configuration.
        Returns list of warning messages.
        """
        warnings = []

        if self.config.temperature > 0.5:
            warnings.append("Высокая температура может привести к галлюцинациям")

        if self.config.max_tokens > 4096:
            warnings.append("Слишком большой лимит токенов может привести к избыточному генерации")

        if self.config.repetition_penalty < 1.1:
            warnings.append("Низкий penalty за повторы может привести к зацикливанию")

        if self.config.early_stopping is False:
            warnings.append("Отключен early stopping - модель может генерировать бесконечно")

        return warnings
