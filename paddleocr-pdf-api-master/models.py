import os

_import_errors = []

# Robust import for PaddleOCR, draw_ocr and PPStructureV3
try:
    from paddleocr import PaddleOCR
except ImportError:
    try:
        from paddleocr.paddleocr import PaddleOCR
    except ImportError:
        PaddleOCR = None

try:
    from paddleocr import draw_ocr
except ImportError:
    try:
        from paddleocr.tools.infer.utility import draw_ocr
    except ImportError:
        draw_ocr = None

try:
    import paddleocr
    from paddleocr import PPStructureV3
except Exception as e:
    err_msg = f"Failed to import PPStructureV3 from paddleocr: {type(e).__name__}: {e}"
    _import_errors.append(err_msg)
    PPStructureV3 = None

# Fallback for draw_ocr if still None
if draw_ocr is None:
    def draw_ocr(image, boxes, txts=None, scores=None, font_path=None, **kwargs):
        return image

# Provide draw_OCR as an alias for compatibility
draw_OCR = draw_ocr

class ModelLoader:
    _ocr_model = None
    _layout_engine = None

    @classmethod
    def load_ocr_model(cls):
        if cls._ocr_model is None:
            if PaddleOCR is None:
                raise ImportError("PaddleOCR could not be imported. Please check installation and python path.")
            print("Loading PaddleOCR model...")
            try:
                cls._ocr_model = PaddleOCR(
                    use_angle_cls=True,
                    use_doc_orientation_classify=True,
                    use_doc_unwarping=True,
                    lang='ru'
                )
                print("Model loaded successfully.")
            except Exception as e:
                print(f"Error during PaddleOCR initialization: {e}")
                import traceback
                traceback.print_exc()
                raise
        return cls._ocr_model

    @classmethod
    def load_layout_engine(cls):
        if cls._layout_engine is None:
            if PPStructureV3 is None:
                print("Error: PPStructureV3 is not available (failed to import at startup)")
                return None
            print("Initializing PPStructureV3 layout engine...")
            try:
                cls._layout_engine = PPStructureV3(
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_seal_recognition=False,
                    use_table_recognition=False,
                    use_formula_recognition=False,
                    use_chart_recognition=False,
                    use_region_detection=False,
                )
                print("PPStructureV3 initialized")
            except Exception as e:
                print(f"Error initializing PPStructureV3: {e}")
                import traceback
                traceback.print_exc()
                cls._layout_engine = None
        return cls._layout_engine
