import json
import os
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
from PIL import Image, ImageDraw

from config import UPLOAD_DIR, DPI
from database import get_db
from models import ModelLoader, draw_ocr
from ocr_logic import (
    normalize_ocr_result, extract_line_text, extract_line_data,
    extract_text, extract_structured
)
from utils import convert_html_tables

class OCRWorker:
    def __init__(self):
        self._thread = None
        self._stop = threading.Event()
        self._cancelled: set[str] = set()
        self._cancel_lock = threading.Lock()

    def cancel_job(self, job_id: str):
        with self._cancel_lock:
            self._cancelled.add(job_id)

    def _is_cancelled(self, job_id: str) -> bool:
        with self._cancel_lock:
            return job_id in self._cancelled

    def _clear_cancelled(self, job_id: str):
        with self._cancel_lock:
            self._cancelled.discard(job_id)

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _run(self):
        retry_count = 0
        max_retries = 5
        retry_delay = 5  # seconds
        
        while not self._stop.is_set():
            try:
                ocr = ModelLoader.load_ocr_model()
                # Model loaded successfully - process jobs
                while not self._stop.is_set():
                    job = self._pick_next_job()
                    if job is None:
                        time.sleep(1)
                        continue
                    try:
                        self._process_job(ocr, job)
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        with get_db() as db:
                            db.execute(
                                "UPDATE jobs SET status = 'failed', error = ?, updated_at = ? WHERE id = ?",
                                (str(e), time.time(), job["id"]),
                            )
            except Exception as e:
                retry_count += 1
                if retry_count > max_retries:
                    print(f"Max retries ({max_retries}) exceeded for model loading. Worker stopping.")
                    return
                print(f"Failed to load OCR model (attempt {retry_count}/{max_retries}): {e}")
                import traceback
                traceback.print_exc()
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                # Increase delay for next retry
                retry_delay = min(retry_delay * 2, 60)

    def _pick_next_job(self):
        with get_db() as db:
            row = db.execute(
                "SELECT id, filename, detect_seal FROM jobs WHERE status = 'queued' ORDER BY created_at LIMIT 1"
            ).fetchone()
            if row:
                now = time.time()
                db.execute(
                    "UPDATE jobs SET status = 'processing', updated_at = ? WHERE id = ?",
                    (now, row["id"]),
                )
                return dict(row)
        return None

    def _process_job(self, ocr, job):
        job_id = job["id"]
        job_dir = Path(UPLOAD_DIR) / job_id
        input_candidates = list(job_dir.glob("input.*"))
        if not input_candidates:
            raise FileNotFoundError(f"no input file in {job_dir}")
        input_path = input_candidates[0]
        is_pdf = input_path.suffix.lower() == ".pdf"

        if is_pdf:
            pdf = pdfium.PdfDocument(str(input_path))
            total_pages = len(pdf)
        else:
            pdf = None
            total_pages = 1

        with get_db() as db:
            db.execute(
                "UPDATE jobs SET total_pages = ?, updated_at = ? WHERE id = ?",
                (total_pages, time.time(), job_id),
            )

        for page_idx in range(total_pages):
            if self._stop.is_set():
                return
            if self._is_cancelled(job_id):
                self._handle_cancel(job_id, page_idx, total_pages)
                return

            pil_image = self._render_page(pdf, input_path, page_idx, is_pdf)
            
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                pil_image.save(tmp.name)
                tmp_path = tmp.name

            try:
                mask_count = 0
                if job.get("detect_seal"):
                    mask_count = self._handle_seal_detection(job_id, page_idx, pil_image, tmp_path, job_dir)

                # Run OCR
                result = ocr.ocr(tmp_path)
                normalized_pages = normalize_ocr_result(result)

                if not normalized_pages or all(not p for p in normalized_pages):
                    print(f"[{job_id[:8]}] Warning: No text found on page {page_idx + 1}")

                self._log_ocr_text(job_id, page_idx, normalized_pages)
                self._visualize_results(job_id, page_idx, pil_image, normalized_pages, job_dir)

                page_markdown = extract_text(normalized_pages)
                page_markdown = convert_html_tables(page_markdown)
                structured_data = extract_structured(normalized_pages, page_num=page_idx + 1)

                self._save_page_results(job_id, page_idx, page_markdown, structured_data, job_dir)
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        self._finalize_job(job_id, total_pages, job_dir)

    def _handle_cancel(self, job_id, page_idx, total_pages):
        self._clear_cancelled(job_id)
        with get_db() as db:
            db.execute(
                "UPDATE jobs SET status = 'cancelled', updated_at = ? WHERE id = ?",
                (time.time(), job_id),
            )
        print(f"[{job_id[:8]}] Job cancelled at page {page_idx + 1}/{total_pages}")

    def _render_page(self, pdf, input_path, page_idx, is_pdf):
        scale = DPI / 72
        if is_pdf:
            page = pdf[page_idx]
            bitmap = page.render(scale=scale)
            return bitmap.to_pil()
        else:
            pil_image = Image.open(input_path)
            if pil_image.mode not in ("RGB", "L"):
                pil_image = pil_image.convert("RGB")
            return pil_image

    def _handle_seal_detection(self, job_id, page_idx, pil_image, tmp_path, job_dir):
        mask_count = 0
        try:
            print(f"[{job_id[:8]}] Starting seal detection on page {page_idx + 1}...")
            layout_engine = ModelLoader.load_layout_engine()
            if layout_engine:
                output = layout_engine.predict(input=tmp_path)
                layout_result = next(iter(output), None)

                boxes = []
                if layout_result is not None:
                    if hasattr(layout_result, 'layout_det_res') and layout_result.layout_det_res:
                        lres = layout_result.layout_det_res
                        boxes = lres.get('boxes', []) if isinstance(lres, dict) else getattr(lres, 'boxes', [])
                    elif isinstance(layout_result, dict):
                        boxes = layout_result.get('boxes', []) or layout_result.get('layout_det_res', {}).get('boxes', [])

                if boxes:
                    seals_dir = job_dir / "seals"
                    seals_dir.mkdir(exist_ok=True)
                    draw = ImageDraw.Draw(pil_image)

                    for i, region in enumerate(boxes):
                        label = (region.get('label', '') if hasattr(region, 'get') else getattr(region, 'label', '')).lower()
                        coord = (region.get('coordinate', region.get('bbox', [0, 0, 0, 0])) if hasattr(region, 'get') 
                                 else getattr(region, 'coordinate', getattr(region, 'bbox', [0, 0, 0, 0])))

                        if label == 'seal':
                            x1, y1, x2, y2 = map(int, coord)
                            seal_crop = pil_image.crop((x1, y1, x2, y2))
                            seal_path = seals_dir / f"page_{page_idx + 1}_seal_{mask_count}.png"
                            seal_crop.save(seal_path)
                            draw.rectangle([x1, y1, x2, y2], fill="white")
                            mask_count += 1
                    
                    if mask_count > 0:
                        pil_image.save(tmp_path)
        except Exception as e:
            print(f"[{job_id[:8]}] Seal detection failed: {e}")
        return mask_count

    def _log_ocr_text(self, job_id, page_idx, normalized_pages):
        try:
            log_texts = []
            for page_lines in normalized_pages:
                if not page_lines: continue
                for line in page_lines:
                    t = extract_line_text(line)
                    if t: log_texts.append(t)
            if log_texts:
                out = " | ".join(log_texts[:20])
                if len(log_texts) > 20: out += f" ... (+{len(log_texts) - 20} lines)"
                print(f"[{job_id[:8]}] OCR text: {out[:500]}")
        except Exception as e:
            print(f"[{job_id[:8]}] Logging text failed: {e}")

    def _visualize_results(self, job_id, page_idx, pil_image, normalized_pages, job_dir):
        try:
            vis_dir = job_dir / "visualized"
            vis_dir.mkdir(exist_ok=True)
            vis_path = vis_dir / f"page_{page_idx + 1}.png"

            if normalized_pages and normalized_pages[0]:
                boxes, texts, scores = [], [], []
                for line in normalized_pages[0]:
                    bbox, text, score = extract_line_data(line)
                    if text:
                        boxes.append(bbox)
                        texts.append(text)
                        scores.append(score)

                if boxes and texts and scores:
                    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
                    if not os.path.exists(font_path): font_path = None
                    im_show = draw_ocr(np.array(pil_image), boxes, texts, scores, font_path=font_path)
                    Image.fromarray(im_show).save(str(vis_path))
                else:
                    pil_image.save(str(vis_path))
            else:
                pil_image.save(str(vis_path))
        except Exception as e:
            print(f"[{job_id[:8]}] Visualization failed: {e}")
            try: pil_image.save(str(vis_path))
            except: pass

    def _save_page_results(self, job_id, page_idx, page_markdown, structured_data, job_dir):
        (job_dir / f"page_{page_idx + 1}.md").write_text(page_markdown, encoding="utf-8")
        (job_dir / f"page_{page_idx + 1}.json").write_text(
            json.dumps(structured_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        now = time.time()
        with get_db() as db:
            db.execute(
                "INSERT INTO pages (job_id, page_num, markdown, result_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (job_id, page_idx + 1, page_markdown, json.dumps(structured_data, ensure_ascii=False), now),
            )
            db.execute(
                "UPDATE jobs SET processed_pages = ?, updated_at = ? WHERE id = ?",
                (page_idx + 1, now, job_id),
            )
        print(f"[{job_id[:8]}] Page {page_idx + 1} done")

    def _finalize_job(self, job_id, total_pages, job_dir):
        with get_db() as db:
            db.execute(
                "UPDATE jobs SET status = 'completed', updated_at = ? WHERE id = ?",
                (time.time(), job_id),
            )
            pages_data = db.execute(
                "SELECT page_num, markdown, result_json FROM pages WHERE job_id = ? ORDER BY page_num",
                (job_id,)
            ).fetchall()

        full_markdown = ""
        full_structured = []
        for p in pages_data:
            full_markdown += f"# Страница {p['page_num']}\n\n{p['markdown']}\n\n---\n\n"
            if p['result_json']:
                full_structured.extend(json.loads(p['result_json']))

        (job_dir / "result.md").write_text(full_markdown, encoding="utf-8")
        (job_dir / "result.json").write_text(
            json.dumps(full_structured, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[{job_id[:8]}] Job completed ({total_pages} pages).")
