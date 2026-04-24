import os
import time
import logging
import requests
import uuid
from flask import Flask, request, render_template, send_file, flash, redirect, url_for, session, jsonify
from werkzeug.utils import secure_filename
from datetime import timedelta
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff', 'webp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 64 MB limit
app.secret_key = os.environ.get('SECRET_KEY', 'default-secret-key')
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=1)

PADDLEOCR_API_URL = os.environ.get('PADDLEOCR_API_URL', 'http://paddleocr-api:8000')

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('Файл не найден', 'error')
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        flash('Файл не выбран', 'error')
        return redirect(url_for('index'))

    if not allowed_file(file.filename):
        flash('Неверный тип файла.', 'error')
        return redirect(url_for('index'))

    filename = secure_filename(file.filename)
    
    # Send file to PaddleOCR API
    try:
        files = {'file': (filename, file.read(), file.content_type)}
        response = requests.post(f"{PADDLEOCR_API_URL}/ocr", files=files)
        response.raise_for_status()
        job_data = response.json()
        
        task_id = job_data['job_id']
        session['task_id'] = task_id
        session['filename'] = filename
        
        return redirect(url_for('status', task_id=task_id))
    except Exception as e:
        logger.error(f"Error sending file to PaddleOCR API: {e}")
        flash(f"Ошибка подключения к OCR бэкенду: {e}", 'error')
        return redirect(url_for('index'))

@app.route('/status/<task_id>')
def status(task_id):
    return render_template('status.html', task_id=task_id)

@app.route('/api/task_status/<task_id>')
def task_status(task_id):
    try:
        response = requests.get(f"{PADDLEOCR_API_URL}/ocr/{task_id}")
        response.raise_for_status()
        job_data = response.json()
        
        # Map PaddleOCR API status to what frontend expects
        # Frontend expects: status, progress, step, redirect, error
        status_map = {
            'queued': 'processing',
            'processing': 'processing',
            'completed': 'completed',
            'failed': 'failed',
            'cancelled': 'failed'
        }
        
        status = status_map.get(job_data['status'], 'processing')
        
        progress = 0
        if job_data['total_pages'] > 0:
            progress = int((job_data['processed_pages'] / job_data['total_pages']) * 100)
        elif job_data['status'] == 'completed':
            progress = 100
            
        result = {
            "status": status,
            "progress": progress,
            "step": "ocr" if status == "processing" else "",
            "total_pages": job_data['total_pages'],
            "processed_pages": job_data['processed_pages']
        }
        
        if status == 'completed':
            result["redirect"] = url_for('success', task_id=task_id)
            # Calculate processing time
            if 'created_at' in job_data and 'updated_at' in job_data:
                processing_time = job_data['updated_at'] - job_data['created_at']
                session[f'processing_time_{task_id}'] = round(processing_time, 2)
        
        if status == 'failed':
            result["error"] = job_data.get('error', 'Неизвестная ошибка')
            result["redirect"] = url_for('index')
            
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error checking status: {e}")
        return jsonify({"status": "failed", "error": str(e), "redirect": url_for('index')})

@app.route('/success/<task_id>')
def success(task_id):
    filename = session.get('filename', 'документ')
    processing_time = session.get(f'processing_time_{task_id}', 'Н/Д')
    return render_template('success.html', task_id=task_id, filename=filename, processing_time=processing_time)

@app.route('/api/image/<task_id>/<int:page_num>')
def get_image(task_id, page_num):
    try:
        response = requests.get(f"{PADDLEOCR_API_URL}/ocr/{task_id}/image/{page_num}", stream=True)
        response.raise_for_status()
        return send_file(response.raw, mimetype='image/png')
    except Exception as e:
        logger.error(f"Error fetching image: {e}")
        return "Изображение не найдено", 404

@app.route('/download_result/<task_id>')
def download_result(task_id):
    try:
        response = requests.get(f"{PADDLEOCR_API_URL}/ocr/{task_id}/result")
        response.raise_for_status()
        result_data = response.json()
        
        # Extract full markdown
        full_markdown = ""
        for page in result_data.get('pages', []):
            full_markdown += f"# Страница {page['page_num']}\n\n{page['markdown']}\n\n---\n\n"
            
        from io import BytesIO
        mem = BytesIO()
        mem.write(full_markdown.encode('utf-8'))
        mem.seek(0)
        
        return send_file(
            mem,
            mimetype='text/markdown',
            as_attachment=True,
            download_name=f"ocr_result_{task_id}.md"
        )
    except Exception as e:
        logger.error(f"Error downloading result: {e}")
        flash(f"Ошибка при скачивании результата: {e}", 'error')
        return redirect(url_for('index'))

@app.route('/new_conversion')
def new_conversion():
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    host = '0.0.0.0'
    port = int(os.environ.get('PORT', 8011))
    app.run(debug=True, host=host, port=port)
