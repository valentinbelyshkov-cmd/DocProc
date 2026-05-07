"""
Test script for seal detection functionality.
Run: python test_seal_detector.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from models.seal_detector import YOLOSealDetector, SealDetectorFactory, get_seal_detector
from PIL import Image


def test_direct_yolo():
    """Test YOLOSealDetector directly."""
    print("=" * 50)
    print("Test 1: Direct YOLOSealDetector")
    print("=" * 50)

    detector = YOLOSealDetector(
        model_path='weights/best.pt',
        confidence_threshold=0.3
    )

    print(f"Detector loaded: {detector.is_available()}")
    assert detector.is_available(), "Detector should be available"

    img = Image.open('2.jpg')
    results = detector.detect(img)

    print(f"Detected {len(results)} seal(s)")
    for i, r in enumerate(results):
        print(f"  Seal {i+1}: bbox={r.bbox}, conf={r.confidence:.2f}, type={r.seal_type}")

    assert len(results) > 0, "Should detect at least one seal"
    print("✓ Test passed!")


def test_factory():
    """Test SealDetectorFactory."""
    print("\n" + "=" * 50)
    print("Test 2: SealDetectorFactory")
    print("=" * 50)

    detector = SealDetectorFactory.get_best_available('weights/best.pt')
    print(f"Factory detector loaded: {detector.is_available()}")

    img = Image.open('2.jpg')
    results = detector.detect(img)
    print(f"Detected {len(results)} seal(s)")

    assert len(results) > 0, "Should detect at least one seal"
    print("✓ Test passed!")


def test_global_detector():
    """Test global get_seal_detector function."""
    print("\n" + "=" * 50)
    print("Test 3: Global get_seal_detector function")
    print("=" * 50)

    detector = get_seal_detector(model_path='weights/best.pt')
    print(f"Global detector loaded: {detector.is_available()}")

    img = Image.open('2.jpg')
    results = detector.detect(img)
    print(f"Detected {len(results)} seal(s)")

    assert len(results) > 0, "Should detect at least one seal"
    print("✓ Test passed!")


def test_no_seals():
    """Test image without seals."""
    print("\n" + "=" * 50)
    print("Test 4: Image without seals")
    print("=" * 50)

    detector = YOLOSealDetector(model_path='weights/best.pt', confidence_threshold=0.5)
    img = Image.open('test_ocr.png')  # Small test image without seals
    results = detector.detect(img)
    print(f"Detected {len(results)} seal(s)")

    assert len(results) == 0, "Should detect no seals in test_ocr.png"
    print("✓ Test passed!")


def test_filter_and_utils():
    """Test filter_by_confidence and get_largest_seal utilities."""
    print("\n" + "=" * 50)
    print("Test 5: Utility functions")
    print("=" * 50)

    detector = YOLOSealDetector(model_path='weights/best.pt', confidence_threshold=0.3)
    img = Image.open('2.jpg')
    results = detector.detect(img)

    # Filter by confidence
    high_conf = detector.filter_by_confidence(results, min_confidence=0.9)
    print(f"High confidence (>=0.9): {len(high_conf)}")

    # Get largest seal
    largest = detector.get_largest_seal(results)
    if largest:
        print(f"Largest seal: {largest.bbox}")

    print("✓ Test passed!")


if __name__ == "__main__":
    print("Testing seal detection with weights/best.pt\n")

    try:
        test_direct_yolo()
        test_factory()
        test_global_detector()
        test_no_seals()
        test_filter_and_utils()

        print("\n" + "=" * 50)
        print("All tests passed! ✓")
        print("=" * 50)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
