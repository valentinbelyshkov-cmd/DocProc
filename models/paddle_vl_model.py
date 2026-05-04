"""
PaddleOCR-VL model implementation.
"""
from typing import Optional, Dict, Any
import requests
import logging
from io import BytesIO
import PIL.Image

from models.base_model import BaseModel, ModelConfig, GenerationResult
import config as app_config

logger = logging.getLogger(__name__)


class PaddleOCRVLModel(BaseModel):
    """
    PaddleOCR-VL-1.5 model wrapper.
    Hosted vision OCR model.
    """

    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        config: Optional[ModelConfig] = None
    ):
        super().__init__(config or ModelConfig.for_ocr())
        self.endpoint_url = endpoint_url or app_config.PADDLEOCR_VL_ENDPOINT
        self.name = "paddle-vl"

    def generate(
        self,
        prompt: str,
        image: Optional[PIL.Image.Image] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> GenerationResult:
        """
        Generate using PaddleOCR-VL endpoint.
        Note: PaddleOCR-VL typically works with images only,
        prompt is passed as additional instruction.
        """
        if not self.endpoint_url:
            return GenerationResult(
                content="",
                error="PaddleOCR-VL endpoint not configured"
            )

        if not image:
            return GenerationResult(
                content="",
                error="Image is required for PaddleOCR-VL"
            )

        buffered = BytesIO()
        image.save(buffered, format="PNG")
        files = {'image': ('image.png', buffered.getvalue(), 'image/png')}

        # Additional parameters
        data = {}
        if prompt:
            data['instruction'] = prompt

        try:
            logger.info(f"PaddleOCR-VL request: {self.endpoint_url}")
            response = requests.post(
                self.endpoint_url,
                files=files,
                data=data,
                timeout=self.config.timeout
            )
            response.raise_for_status()

            result = response.json()
            return GenerationResult(
                content=result.get('text', ''),
                raw_response=result,
                model_name="paddle-vl"
            )

        except requests.exceptions.Timeout:
            return GenerationResult(content="", error="Таймаут запроса к PaddleOCR-VL")
        except requests.exceptions.RequestException as e:
            logger.error(f"PaddleOCR-VL request failed: {e}")
            return GenerationResult(content="", error=f"Ошибка PaddleOCR-VL: {str(e)}")
        except Exception as e:
            return GenerationResult(content="", error=f"Неизвестная ошибка: {str(e)}")

    def extract_text_and_tables(
        self,
        image: PIL.Image.Image,
        custom_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract text and tables using PaddleOCR-VL."""
        result = self.generate(custom_prompt or "", image=image)

        if not result.success:
            return {
                'text': f"Ошибка: {result.error}",
                'tables': [],
                'raw': None
            }

        # Parse response if it contains JSON
        try:
            import json
            parsed = json.loads(result.content)
            return {
                'text': parsed.get('text', result.content),
                'tables': parsed.get('tables', []),
                'raw': parsed
            }
        except:
            return {
                'text': result.content,
                'tables': [],
                'raw': None
            }