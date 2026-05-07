"""
Factory for AI models.
Provides a registry of all available models.
"""
from typing import Dict, Type, Optional, List
from models.base_model import BaseModel, ModelConfig
from models.openrouter_model import OpenRouterModel, OpenRouterClaudeModel, OpenRouterGeminiModel
from models.glm_model import GLMOCRModel
from models.ollama_model import OllamaModel, NoctrixLightOnOCRModel
from models.lighton_ocr_model import LightOnOCRModel
import config as app_config


class ModelRegistry:
    """
    Registry for AI models.
    Manages registration and retrieval of models for OCR tasks.
    """

    _models: Dict[str, Type[BaseModel]] = {}

    @classmethod
    def register(cls, name: str, model_class: Type[BaseModel]) -> None:
        """Register a new model."""
        if not issubclass(model_class, BaseModel):
            raise TypeError(f"{model_class} must inherit from BaseModel")
        cls._models[name] = model_class

    @classmethod
    def create(cls, name: str, **kwargs) -> Optional[BaseModel]:
        """Create an instance of the specified model."""
        model_class = cls._models.get(name)
        if model_class:
            return model_class(**kwargs)
        return None

    @classmethod
    def get_available_models(cls) -> List[str]:
        """Get list of all registered model names."""
        return list(cls._models.keys())

    @classmethod
    def get_default_model(cls) -> BaseModel:
        """Get the default model based on configuration."""
        # Priority: OpenRouter > GLM > Ollama
        if app_config.OPENROUTER_API_KEY:
            return OpenRouterModel()

        if app_config.ZHIPUAI_API_KEY:
            return GLMOCRModel()

        if app_config.OLLAMA_BASE_URL:
            return OllamaModel()

        # Return OpenRouter by default (most widely compatible)
        return OpenRouterModel()

    @classmethod
    def get_model_info(cls, name: str) -> Optional[Dict]:
        """Get information about a model."""
        model_class = cls._models.get(name)
        if not model_class:
            return None

        # Create temporary instance to get default config
        temp_model = model_class()
        return {
            'name': name,
            'display_name': cls._get_display_name(name),
            'supports_vision': True,
            'config': {
                'temperature': temp_model.config.temperature,
                'max_tokens': temp_model.config.max_tokens,
            }
        }

    @classmethod
    def _get_display_name(cls, name: str) -> str:
        """Get human-readable name for model."""
        names = {
            'openrouter': 'OpenRouter (Gemini)',
            'openrouter-claude': 'OpenRouter (Claude)',
            'openrouter-gemini': 'Google Gemini via OpenRouter',
            'glm': 'ZhipuAI GLM-4V',
            'ollama': 'Ollama (Local)',
            'ollama-glm': 'Ollama GLM-OCR',
            'noctrix': 'Noctrix LightOnOCR',
            'lightonocr': 'LightOnOCR-2 (Ollama)',
        }
        return names.get(name, name)


# Register all built-in models
ModelRegistry.register('openrouter', OpenRouterModel)
ModelRegistry.register('openrouter-claude', OpenRouterClaudeModel)
ModelRegistry.register('openrouter-gemini', OpenRouterGeminiModel)
ModelRegistry.register('glm', GLMOCRModel)
ModelRegistry.register('ollama', OllamaModel)
ModelRegistry.register('ollama-glm', OllamaModel)
ModelRegistry.register('noctrix', NoctrixLightOnOCRModel)
ModelRegistry.register('lightonocr', LightOnOCRModel)