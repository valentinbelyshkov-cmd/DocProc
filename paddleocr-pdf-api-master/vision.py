import base64
import io
import os
import threading
import time
from PIL import Image
from config import (
    IMAGE_DESCRIPTION_PROVIDER, IMAGE_DESCRIPTION_API_URL, IMAGE_DESCRIPTION_API_KEY,
    IMAGE_DESCRIPTION_API_VERSION, IMAGE_DESCRIPTION_TIMEOUT, IMAGE_DESCRIPTION_API_MODE,
    IMAGE_DESCRIPTION_MODEL, IMAGE_DESCRIPTION_MAX_EDGE_PX, IMAGE_DESCRIPTION_MAX_RETRIES,
    HTML_IMG_RE, MD_IMG_RE, IMG_PATH_RE, IMAGE_DESCRIPTION_PROMPT_OVERRIDES,
    IMAGE_DESCRIPTION_DEFAULT_PROMPT, NATIVE_RENDERED_LABELS, IMAGE_DESCRIPTION_LABELS,
    IMAGE_DESCRIPTION_MIN_PIXELS, IMAGE_DESCRIPTION_MAX_PER_PAGE, IMAGE_DESCRIPTION_ON_ERROR
)
from utils import strip_image_tags

class VisionClient:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(VisionClient, cls).__new__(cls)
                cls._instance._client = cls._instance._build_client()
        return cls._instance

    def _build_client(self):
        from openai import AzureOpenAI, OpenAI

        if IMAGE_DESCRIPTION_PROVIDER == "azure":
            return AzureOpenAI(
                azure_endpoint=IMAGE_DESCRIPTION_API_URL,
                api_key=IMAGE_DESCRIPTION_API_KEY or "none",
                api_version=IMAGE_DESCRIPTION_API_VERSION,
                timeout=IMAGE_DESCRIPTION_TIMEOUT,
            )
        return OpenAI(
            base_url=IMAGE_DESCRIPTION_API_URL,
            api_key=IMAGE_DESCRIPTION_API_KEY or "none",
            timeout=IMAGE_DESCRIPTION_TIMEOUT,
        )

    def _encode_image(self, pil_image) -> str:
        img = pil_image
        if IMAGE_DESCRIPTION_MAX_EDGE_PX > 0 and max(img.size) > IMAGE_DESCRIPTION_MAX_EDGE_PX:
            img = img.copy()
            img.thumbnail((IMAGE_DESCRIPTION_MAX_EDGE_PX, IMAGE_DESCRIPTION_MAX_EDGE_PX))
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    def _vision_call(self, data_url: str, prompt: str) -> str:
        if IMAGE_DESCRIPTION_API_MODE == "responses":
            resp = self._client.responses.create(
                model=IMAGE_DESCRIPTION_MODEL,
                input=[{"role": "user", "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": data_url},
                ]}],
                timeout=IMAGE_DESCRIPTION_TIMEOUT,
            )
            return (resp.output_text or "").strip()
        resp = self._client.chat.completions.create(
            model=IMAGE_DESCRIPTION_MODEL,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]}],
            timeout=IMAGE_DESCRIPTION_TIMEOUT,
        )
        return (resp.choices[0].message.content or "").strip()

    def describe_one(self, pil_image, prompt: str) -> str:
        data_url = self._encode_image(pil_image)
        last_err = None
        for attempt in range(IMAGE_DESCRIPTION_MAX_RETRIES + 1):
            try:
                return self._vision_call(data_url, prompt)
            except Exception as e:
                last_err = e
                if attempt < IMAGE_DESCRIPTION_MAX_RETRIES:
                    time.sleep(min(2 ** attempt, 5))
        raise last_err

def _parse_image_path(path: str):
    name = os.path.basename(path)
    m = IMG_PATH_RE.search(name)
    if not m:
        return None
    label = m.group("label").lower()
    x1, y1, x2, y2 = (int(m.group(i)) for i in (2, 3, 4, 5))
    return label, (x1, y1, x2, y2)

def _prompt_for(label: str) -> str:
    return IMAGE_DESCRIPTION_PROMPT_OVERRIDES.get(label.lower(), IMAGE_DESCRIPTION_DEFAULT_PROMPT)

def _replace_image_tags(text: str, replacements: dict) -> str:
    def _sub(match):
        src = match.group("src")
        key = _match_replacement_key(src, replacements)
        if key is None:
            return ""
        return replacements[key]

    text = HTML_IMG_RE.sub(_sub, text)
    text = MD_IMG_RE.sub(_sub, text)
    import re
    text = re.sub(r'<div[^>]*>\s*</div>', "", text)
    text = re.sub(r'\n{3,}', "\n\n", text)
    return text

def _match_replacement_key(src: str, replacements: dict):
    if src in replacements:
        return src
    base = os.path.basename(src)
    for key in replacements:
        if os.path.basename(key) == base:
            return key
    return None

def describe_images(text: str, images: dict, page_num: int = 0, job_id: str = "") -> str:
    if not text or not images:
        return strip_image_tags(text)

    referenced = set()
    for m in HTML_IMG_RE.finditer(text):
        referenced.add(m.group("src"))
    for m in MD_IMG_RE.finditer(text):
        referenced.add(m.group("src"))

    client = VisionClient()
    replacements: dict = {}
    described = 0

    for path, pil_image in images.items():
        if path not in referenced:
            base = os.path.basename(path)
            if not any(os.path.basename(r) == base for r in referenced):
                continue

        parsed = _parse_image_path(path)
        if parsed is None:
            replacements[path] = ""
            continue
        label, (x1, y1, x2, y2) = parsed

        if label in NATIVE_RENDERED_LABELS:
            continue
        if label not in IMAGE_DESCRIPTION_LABELS:
            replacements[path] = ""
            continue

        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area < IMAGE_DESCRIPTION_MIN_PIXELS:
            replacements[path] = ""
            continue

        if described >= IMAGE_DESCRIPTION_MAX_PER_PAGE:
            replacements[path] = ""
            continue

        prompt = _prompt_for(label)
        label_display = label.replace("_", " ").title()

        try:
            desc = client.describe_one(pil_image, prompt)
        except Exception as e:
            print(f"[image-desc] job={job_id[:8]} page={page_num} label={label} error: {e}")
            if IMAGE_DESCRIPTION_ON_ERROR == "fail":
                raise
            if IMAGE_DESCRIPTION_ON_ERROR == "placeholder":
                replacements[path] = f"> **[{label_display}]** [image description unavailable]"
            else:
                replacements[path] = ""
            continue

        if not desc:
            replacements[path] = ""
            continue

        replacements[path] = f"> **[{label_display}]** {desc}"
        described += 1

    return _replace_image_tags(text, replacements)
