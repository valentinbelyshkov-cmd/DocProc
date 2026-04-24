try:
    from paddleocr import PaddleOCR
except ImportError:
    try:
        from paddleocr.paddleocr import PaddleOCR
    except ImportError:
        print("Error: Could not import PaddleOCR. Please install paddleocr package.")
        import sys
        sys.exit(1)

import cv2
import json

# Инициализация с включённым документным пайплайном
ocr = PaddleOCR(
    use_angle_cls=True,            # Коррекция наклона текста
    use_doc_orientation_classify=True,  # Определение ориентации страницы
    use_doc_unwarping=True,        # Выравнивание искривлённых страниц
    lang='ru',                     # 'ru', 'en', 'ch', 'multi' (мультиязычный)
)

# Путь к документу (поддерживаются: jpg, png, pdf, tiff, webp)
img_path = "mydoc.pdf"

# Запуск распознавания
result = ocr.predict(img_path)
# Форматирование вывода
structured_output = []
for page_idx, lines in enumerate(result):
    page_data = {"page": page_idx + 1, "blocks": []}
    if lines:
        for word_info in lines:
            bbox = word_info[0]
            raw_info = word_info[1]

            # 🔍 Универсальный парсинг под все версии PaddleOCR
            if isinstance(raw_info, dict):
                text = raw_info.get("text", "")
                confidence = raw_info.get("confidence", 0.0)
            elif isinstance(raw_info, (list, tuple)) and len(raw_info) == 2:
                text, confidence = raw_info
            else:
                text = str(raw_info)
                confidence = 1.0

            page_data["blocks"].append({
                "text": text,
                "confidence": round(float(confidence), 4),
                "bbox": bbox
            })
    structured_output.append(page_data)

# Сохранение в JSON
with open("ocr_result.json", "w", encoding="utf-8") as f:
    json.dump(structured_output, f, ensure_ascii=False, indent=2)

print("✅ Обработка завершена. Результат сохранён в ocr_result.json")