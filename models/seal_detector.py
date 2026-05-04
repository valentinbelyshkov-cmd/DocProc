"""
Seal (stamp) detector module.
Provides interface for seal detection using YOLO models.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any, Dict
import logging
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SealDetectionResult:
    """Result of seal detection."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    seal_type: str = "unknown"
    cropped_image: Optional[Any] = None


class BaseSealDetector(ABC):
    """
    Abstract base class for seal detectors.
    Implementations should use YOLO or other object detection models.
    """

    @abstractmethod
    def detect(self, image: Any) -> List[SealDetectionResult]:
        """
        Detect seals in the image.
        Returns list of detection results with bounding boxes.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the detector is loaded and available."""
        pass

    def filter_by_confidence(
        self,
        results: List[SealDetectionResult],
        min_confidence: float = 0.5
    ) -> List[SealDetectionResult]:
        """Filter detection results by confidence threshold."""
        return [r for r in results if r.confidence >= min_confidence]

    def get_largest_seal(
        self,
        results: List[SealDetectionResult]
    ) -> Optional[SealDetectionResult]:
        """Get the largest detected seal by bounding box area."""
        if not results:
            return None

        def area(r: SealDetectionResult) -> float:
            x1, y1, x2, y2 = r.bbox
            return (x2 - x1) * (y2 - y1)

        return max(results, key=area)


class DummySealDetector(BaseSealDetector):
    """
    Dummy seal detector for testing and fallback.
    Returns empty results.
    """

    def detect(self, image: Any) -> List[SealDetectionResult]:
        """Return empty list (no seals detected)."""
        logger.debug("DummySealDetector: no seals detected")
        return []

    def is_available(self) -> bool:
        """Always return True for dummy detector."""
        return True


class YOLOSealDetector(BaseSealDetector):
    """
    YOLO-based seal detector.
    Uses YOLOv8 or YOLOv5 for seal/stamp detection.

    Model requirements:
    - Trained on seal/stamp dataset
    - Input: image (RGB)
    - Output: bounding boxes with confidence scores
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        device: str = "cpu"
    ):
        """
        Initialize YOLO seal detector.

        Args:
            model_path: Path to YOLO model weights (.pt or .onnx)
            confidence_threshold: Minimum confidence for detections
            device: Device to run inference on ('cpu', 'cuda', 'mps')
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load YOLO model."""
        try:
            from ultralytics import YOLO
            logger.info(f"Loading YOLO seal detector from {self.model_path}")

            if self.model_path:
                self.model = YOLO(self.model_path)
            else:
                # Use default/seeded model
                logger.warning("No model path specified for YOLO seal detector")
                self.model = None

            logger.info("YOLO seal detector loaded successfully")

        except ImportError:
            logger.warning("ultralytics not installed, YOLO seal detector unavailable")
            self.model = None
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self.model = None

    def detect(self, image: Any) -> List[SealDetectionResult]:
        """
        Detect seals using YOLO model.

        Args:
            image: PIL Image or numpy array

        Returns:
            List of SealDetectionResult objects
        """
        if not self.is_available():
            return []

        try:
            # Run inference
            results = self.model.predict(
                image,
                conf=self.confidence_threshold,
                device=self.device,
                verbose=False
            )

            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy().tolist()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])

                    detections.append(SealDetectionResult(
                        bbox=tuple(int(x) for x in xyxy),
                        confidence=conf,
                        seal_type=self._get_class_name(cls)
                    ))

            logger.info(f"YOLO detected {len(detections)} seals")
            return detections

        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            return []

    def is_available(self) -> bool:
        """Check if YOLO model is loaded."""
        return self.model is not None

    def _get_class_name(self, class_id: int) -> str:
        """Map class ID to seal type name."""
        # Example class mapping - adjust based on your training dataset
        class_names = {
            0: "official_seal",
            1: "company_seal",
            2: "stamp",
            3: "round_seal",
            4: "rectangular_seal",
        }
        return class_names.get(class_id, "unknown")


class SealDetectorFactory:
    """
    Factory for creating seal detectors.
    Provides easy access to different detector implementations.
    """

    @staticmethod
    def create(
        detector_type: str = "yolo",
        model_path: Optional[str] = None,
        **kwargs
    ) -> BaseSealDetector:
        """
        Create a seal detector instance.

        Args:
            detector_type: Type of detector ('yolo', 'dummy')
            model_path: Path to model weights (for YOLO)
            **kwargs: Additional arguments for detector

        Returns:
            BaseSealDetector instance
        """
        if detector_type == "yolo":
            return YOLOSealDetector(
                model_path=model_path,
                **kwargs
            )
        elif detector_type == "dummy":
            return DummySealDetector()
        else:
            logger.warning(f"Unknown detector type: {detector_type}, using dummy")
            return DummySealDetector()

    @staticmethod
    def get_best_available(
        model_path: Optional[str] = None
    ) -> BaseSealDetector:
        """
        Get the best available seal detector.
        Tries YOLO first, falls back to dummy.
        """
        try:
            from ultralytics import YOLO
            if model_path:
                return YOLOSealDetector(model_path=model_path)
            return YOLOSealDetector()
        except ImportError:
            logger.info("ultralytics not available, using DummySealDetector")
            return DummySealDetector()


# Global detector instance (lazy loaded)
_seal_detector: Optional[BaseSealDetector] = None


def get_seal_detector(
    detector_type: str = "yolo",
    model_path: Optional[str] = None
) -> BaseSealDetector:
    """
    Get or create global seal detector instance.

    Args:
        detector_type: Type of detector
        model_path: Path to model weights

    Returns:
        BaseSealDetector instance
    """
    global _seal_detector

    if _seal_detector is None:
        _seal_detector = SealDetectorFactory.get_best_available(model_path)

    return _seal_detector


def reload_seal_detector(
    detector_type: str = "yolo",
    model_path: Optional[str] = None
) -> BaseSealDetector:
    """
    Reload the global seal detector.
    Use this to switch detector types or reload model.
    """
    global _seal_detector
    _seal_detector = SealDetectorFactory.create(detector_type, model_path)
    return _seal_detector