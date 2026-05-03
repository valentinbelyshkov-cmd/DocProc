import time
import uuid
import magic
import json
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, Form
from fastapi.responses import FileResponse

from config import (
    UPLOAD_DIR, API_KEY, ALLOWED_IMAGE_EXTS, ALLOWED_IMAGE_MIMES
)
from database import init_db, get_db
from models import PaddleOCR, PPStructureV3, draw_ocr, _import_errors
from worker import OCRWorker

def verify_api_key(request: Request):
    if not API_KEY:
        return
    if request.headers.get("X-API-Key") != API_KEY:
        raise HTTPException(401, "Invalid or missing API key")

worker = OCRWorker()

app = FastAPI(title="PaddleOCR API", version="1.0.0", dependencies=[Depends(verify_api_key)])

@app.on_event("startup")
def startup():
    init_db()
    Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    worker.start()

@app.on_event("shutdown")
def shutdown():
    worker.stop()

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "PaddleOCR": PaddleOCR is not None,
        "PPStructureV3": PPStructureV3 is not None,
        "draw_ocr": draw_ocr is not None,
        "import_errors": _import_errors
    }

@app.post("/ocr")
async def submit_job(file: UploadFile = File(...), detect_seal: Optional[str] = Form(None)):
    is_detect_seal = False
    if detect_seal is not None:
        is_detect_seal = str(detect_seal).lower() in ("true", "1", "on", "yes")
    
    suffix = Path(file.filename or "").suffix.lower()
    is_pdf = suffix == ".pdf"
    is_image = suffix in ALLOWED_IMAGE_EXTS
    if not (is_pdf or is_image):
        raise HTTPException(
            400,
            "Only PDF and image files (PNG, JPG, JPEG, BMP, TIFF, WEBP) are supported",
        )

    content = await file.read()
    mime = magic.from_buffer(content, mime=True)
    if is_pdf and mime != "application/pdf":
        raise HTTPException(400, f"File is not a valid PDF (detected: {mime})")
    if is_image and mime not in ALLOWED_IMAGE_MIMES:
        raise HTTPException(400, f"File is not a valid image (detected: {mime})")

    job_id = uuid.uuid4().hex
    job_dir = Path(UPLOAD_DIR) / job_id
    job_dir.mkdir(parents=True)

    input_path = job_dir / f"input{suffix}"
    input_path.write_bytes(content)

    now = time.time()
    with get_db() as db:
        db.execute(
            "INSERT INTO jobs (id, filename, status, detect_seal, created_at, updated_at) VALUES (?, ?, 'queued', ?, ?, ?)",
            (job_id, file.filename, 1 if is_detect_seal else 0, now, now),
        )

    return {"job_id": job_id, "filename": file.filename, "status": "queued"}

@app.get("/ocr/{job_id}")
def get_job_status(job_id: str):
    with get_db() as db:
        job = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not job:
        raise HTTPException(404, "Job not found")

    return {
        "job_id": job["id"],
        "filename": job["filename"],
        "status": job["status"],
        "detect_seal": job["detect_seal"],
        "total_pages": job["total_pages"],
        "processed_pages": job["processed_pages"],
        "error": job["error"],
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
    }

@app.get("/ocr/{job_id}/image/{page_num}")
def get_job_image(job_id: str, page_num: int):
    job_dir = Path(UPLOAD_DIR) / job_id
    img_path = job_dir / "visualized" / f"page_{page_num}.png"
    if not img_path.exists():
        raise HTTPException(404, "Image not found or not yet processed")
    return FileResponse(str(img_path))

@app.get("/ocr/{job_id}/seals")
def list_job_seals(job_id: str):
    job_dir = Path(UPLOAD_DIR) / job_id
    seals_dir = job_dir / "seals"
    if not seals_dir.exists():
        return {"seals": []}
    
    seals = [f.name for f in sorted(seals_dir.glob("*.png"))]
    return {"seals": seals}

@app.get("/ocr/{job_id}/seals/{filename}")
def get_seal_image(job_id: str, filename: str):
    job_dir = Path(UPLOAD_DIR) / job_id
    seal_path = job_dir / "seals" / filename
    if not seal_path.exists():
        raise HTTPException(404, "Seal not found")
    return FileResponse(str(seal_path))

@app.get("/ocr/{job_id}/pages/{page_num}")
def get_page(job_id: str, page_num: int):
    with get_db() as db:
        job = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if not job:
            raise HTTPException(404, "Job not found")

        page = db.execute(
            "SELECT * FROM pages WHERE job_id = ? AND page_num = ?",
            (job_id, page_num),
        ).fetchone()

    if not page:
        if page_num > job["total_pages"] and job["total_pages"] > 0:
            raise HTTPException(404, f"Page {page_num} does not exist (total: {job['total_pages']})")
        raise HTTPException(202, f"Page {page_num} not yet processed")

    return {
        "job_id": job_id,
        "page_num": page["page_num"],
        "markdown": page["markdown"],
    }

@app.get("/ocr/{job_id}/result")
def get_full_result(job_id: str):
    with get_db() as db:
        job = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if not job:
            raise HTTPException(404, "Job not found")

        pages = db.execute(
            "SELECT page_num, markdown, result_json FROM pages WHERE job_id = ? ORDER BY page_num",
            (job_id,),
        ).fetchall()

    return {
        "job_id": job_id,
        "filename": job["filename"],
        "status": job["status"],
        "total_pages": job["total_pages"],
        "processed_pages": job["processed_pages"],
        "pages": [
            {
                "page_num": p["page_num"],
                "markdown": p["markdown"],
                "result_json": json.loads(p["result_json"]) if p["result_json"] else None
            } for p in pages
        ],
    }

@app.post("/ocr/{job_id}/cancel")
def cancel_job(job_id: str):
    with get_db() as db:
        job = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if not job:
            raise HTTPException(404, "Job not found")
        if job["status"] not in ("queued", "processing"):
            raise HTTPException(400, f"Job cannot be cancelled (status: {job['status']})")
        if job["status"] == "queued":
            db.execute(
                "UPDATE jobs SET status = 'cancelled', updated_at = ? WHERE id = ?",
                (time.time(), job_id),
            )
        else:
            worker.cancel_job(job_id)
    return {"job_id": job_id, "status": "cancelling" if job["status"] == "processing" else "cancelled"}

@app.delete("/ocr/{job_id}")
def delete_job(job_id: str):
    with get_db() as db:
        job = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if not job:
            raise HTTPException(404, "Job not found")
        db.execute("DELETE FROM pages WHERE job_id = ?", (job_id,))
        db.execute("DELETE FROM jobs WHERE id = ?", (job_id,))

    job_dir = Path(UPLOAD_DIR) / job_id
    if job_dir.exists():
        import shutil
        shutil.rmtree(job_dir)

    return {"status": "deleted"}

@app.get("/jobs")
def list_jobs():
    with get_db() as db:
        jobs = db.execute(
            "SELECT id, filename, status, total_pages, processed_pages, created_at FROM jobs ORDER BY created_at DESC"
        ).fetchall()

    return {
        "jobs": [
            {
                "job_id": j["id"],
                "filename": j["filename"],
                "status": j["status"],
                "total_pages": j["total_pages"],
                "processed_pages": j["processed_pages"],
            }
            for j in jobs
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
