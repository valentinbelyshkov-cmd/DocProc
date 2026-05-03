import numpy as np

def normalize_ocr_result(result):
    """Приводит разные форматы ответа PaddleOCR к единому: list of pages,
    где каждая страница — список линий [[bbox, (text, conf)], ...]."""
    if result is None:
        return []

    # Если результат — dict (новый формат PaddleOCR с препроцессорами)
    if isinstance(result, dict):
        return normalize_page_dict(result)

    # Если результат — список
    if isinstance(result, list):
        if not result:
            return []
        first = result[0]
        if first is None:
            return [[]]

        # Список dict'ов — каждый dict это страница
        if isinstance(first, dict):
            pages = []
            for d in result:
                page = normalize_page_dict(d)
                pages.append(page if page else [])
            return pages

        # Старый формат: список линий [[bbox, (text, conf)], ...]
        if isinstance(first, (list, tuple)) and len(first) == 2:
            if isinstance(first[0], (list, tuple, np.ndarray)) and isinstance(first[1], (list, tuple)):
                if len(first[1]) == 2 and isinstance(first[1][0], str):
                    return [result]
            return [result]

        # Вложенный список — одна страница
        if isinstance(first, list):
            return [result]

        return [result]

    return []

def normalize_page_dict(page_dict):
    """Извлекает линии из dict-формата страницы PaddleOCR."""
    if not isinstance(page_dict, dict):
        return []

    polys = page_dict.get('dt_polys', [])
    texts = page_dict.get('rec_texts', [])
    scores = page_dict.get('rec_scores', [])

    if polys and texts:
        page = []
        for i, (poly, text) in enumerate(zip(polys, texts)):
            conf = float(scores[i]) if i < len(scores) else 1.0
            if hasattr(poly, 'tolist'):
                poly = poly.tolist()
            page.append([poly, (text, conf)])
        return page

    # Альтернативные ключи
    if 'texts' in page_dict and 'boxes' in page_dict:
        boxes = page_dict['boxes']
        texts = page_dict['texts']
        scores = page_dict.get('scores', [])
        page = []
        for i, (box, text) in enumerate(zip(boxes, texts)):
            conf = float(scores[i]) if i < len(scores) else 1.0
            if hasattr(box, 'tolist'):
                box = box.tolist()
            page.append([box, (text, conf)])
        return page

    # Вложенные результаты
    for key in ('result', 'ocr_result', 'data', 'page_result'):
        if key in page_dict:
            nested = page_dict[key]
            if isinstance(nested, list):
                norm = normalize_ocr_result(nested)
                return norm[0] if norm else []
            elif isinstance(nested, dict):
                return normalize_page_dict(nested)

    return []

def extract_line_text(line):
    """Достаёт чистый текст из одной линии результата OCR."""
    if not isinstance(line, (list, tuple)) or len(line) < 2:
        return ""
    raw = line[1]
    text = ""
    if isinstance(raw, dict):
        text = raw.get("text", "")
    elif isinstance(raw, (list, tuple)) and len(raw) >= 1:
        text = raw[0]
    elif isinstance(raw, str):
        text = raw

    if isinstance(text, str):
        return text.strip()

    # numpy / bytes fallback
    if hasattr(text, 'item'):
        try:
            text = text.item()
        except:
            pass
    if isinstance(text, np.ndarray) and text.dtype.kind in ('U', 'S'):
        try:
            text = "".join(text.flatten().astype(str).tolist())
        except:
            text = ""
    if isinstance(text, bytes):
        try:
            text = text.decode('utf-8', errors='ignore')
        except:
            text = ""
    return str(text).strip() if text else ""

def extract_line_data(line):
    """Достаёт (bbox, text, score) из линии результата OCR."""
    if not isinstance(line, (list, tuple)) or len(line) < 2:
        return None, "", 1.0

    bbox = line[0]
    if hasattr(bbox, 'tolist'):
        bbox = bbox.tolist()

    raw = line[1]
    text = ""
    score = 1.0

    if isinstance(raw, dict):
        text = raw.get("text", "")
        score = raw.get("confidence", 1.0)
    elif isinstance(raw, (list, tuple)) and len(raw) >= 1:
        text = raw[0]
        score = raw[1] if len(raw) > 1 else 1.0
    elif isinstance(raw, str):
        text = raw

    # Нормализация текста
    if hasattr(text, 'item'):
        try:
            text = text.item()
        except:
            pass
    if isinstance(text, np.ndarray) and text.dtype.kind in ('U', 'S'):
        try:
            text = "".join(text.flatten().astype(str).tolist())
        except:
            text = ""
    if isinstance(text, bytes):
        try:
            text = text.decode('utf-8', errors='ignore')
        except:
            text = ""
    text = str(text).strip() if text else ""

    # Нормализация score
    try:
        if hasattr(score, 'item'):
            score = score.item()
        score = float(score)
    except:
        score = 1.0

    return bbox, text, score

def extract_text(result):
    texts = []
    if not result:
        return ""
    
    # PaddleOCR result is usually a list of pages, each page is a list of lines
    if isinstance(result, list) and len(result) > 0:
        if not isinstance(result[0], list):
            result = [result]

    for page_lines in result:
        if not page_lines:
            continue
        for line in page_lines:
            if not line:
                continue
            
            text = ""
            # Case 1: [bbox, (text, conf)] - PaddleOCR standard
            if isinstance(line, (list, tuple)) and len(line) >= 2:
                raw_info = line[1]
                if isinstance(raw_info, dict):
                    text = raw_info.get("text", "")
                elif isinstance(raw_info, (list, tuple)) and len(raw_info) >= 1:
                    text = raw_info[0]
                else:
                    text = raw_info
            # Case 2: (text, conf) or just text
            elif isinstance(line, (list, tuple)) and len(line) >= 1:
                text = line[0]
            elif isinstance(line, str):
                text = line
            else:
                text = line

            # Handle numpy character arrays or other non-standard string types
            if hasattr(text, 'item') and not isinstance(text, (list, dict, tuple, np.ndarray)):
                try: text = text.item()
                except: pass
            
            if isinstance(text, np.ndarray):
                if text.dtype.kind in ('U', 'S'): # Unicode or String
                    try: text = "".join(text.flatten().astype(str).tolist())
                    except: text = ""
                else:
                    text = "" # Ignore non-textual ndarrays

            if isinstance(text, bytes):
                try: text = text.decode('utf-8', errors='ignore')
                except: text = ""
            
            # Convert to string and check if it's "garbage" technical info
            if text is not None and not isinstance(text, (str, bytes, np.ndarray, list, dict, tuple)):
                text = str(text)

            if isinstance(text, str) and text.strip():
                text_stripped = text.strip()
                # Final safety check against technical strings
                if "shape=" in text_stripped and "dtype=" in text_stripped:
                    continue
                if "<NDARRAY" in text_stripped:
                    continue
                texts.append(text_stripped)
    return "\n\n".join(texts)

def extract_structured(result, page_num=None):
    structured_output = []
    if not result:
        return structured_output
        
    for page_idx, lines in enumerate(result):
        current_page_num = page_num if page_num is not None else page_idx + 1
        page_data = {"page": current_page_num, "blocks": []}
        if lines:
            for word_info in lines:
                if not isinstance(word_info, (list, tuple)) or len(word_info) < 2:
                    continue
                
                bbox = word_info[0]
                # Convert numpy bbox to list
                if hasattr(bbox, "tolist"):
                    bbox = bbox.tolist()
                elif isinstance(bbox, (list, tuple)):
                    bbox = [list(b) if hasattr(b, "tolist") else b for b in bbox]

                raw_info = word_info[1]
                text = ""
                confidence = 1.0
                
                if isinstance(raw_info, dict):
                    text = raw_info.get("text", "")
                    confidence = raw_info.get("confidence", 0.0)
                elif isinstance(raw_info, (list, tuple)):
                    text = raw_info[0] if len(raw_info) >= 1 else ""
                    confidence = raw_info[1] if len(raw_info) >= 2 else 1.0
                elif isinstance(raw_info, str):
                    text = raw_info
                    confidence = 1.0
                else:
                    text = raw_info

                # Handle numpy types for text
                if hasattr(text, 'item') and not isinstance(text, (list, dict, tuple, np.ndarray)):
                    try: text = text.item()
                    except: pass
                if isinstance(text, np.ndarray) and text.dtype.kind in ('U', 'S'):
                    try: text = "".join(text.flatten().astype(str).tolist())
                    except: text = ""
                if isinstance(text, bytes):
                    try: text = text.decode('utf-8', errors='ignore')
                    except: text = ""
                
                if text is not None and not isinstance(text, (str, bytes, np.ndarray, list, dict, tuple)):
                    text = str(text)

                # Ensure confidence is a float
                try:
                    if hasattr(confidence, 'item'):
                        confidence = confidence.item()
                    confidence = float(confidence)
                except:
                    confidence = 0.0

                # Ensure text is string and not technical metadata
                if isinstance(text, str) and text.strip():
                    text_stripped = text.strip()
                    if "shape=" in text_stripped and "dtype=" in text_stripped:
                        continue
                    if "<NDARRAY" in text_stripped:
                        continue
                    
                    page_data["blocks"].append({
                        "text": text_stripped,
                        "confidence": round(float(confidence), 4),
                        "bbox": bbox
                    })
        structured_output.append(page_data)
    return structured_output
