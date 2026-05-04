import os
import logging
import uuid
import json
import zipfile
from io import BytesIO
from flask import Flask, request, render_template, send_file, flash, redirect, url_for, session, jsonify, get_flashed_messages
from werkzeug.utils import secure_filename
from pathlib import Path

import config
from api_client import PaddleOCRClient
from vllm_processor import VLLMProcessor
import converters

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(config)
app.secret_key = config.SECRET_KEY

ocr_client = PaddleOCRClient()
vllm_processor = VLLMProcessor()

def get_processor(task_id):
    if task_id in vllm_processor.tasks:
        return vllm_processor
    return ocr_client

if not os.path.exists(config.UPLOAD_FOLDER):
    os.makedirs(config.UPLOAD_FOLDER)

if not os.path.exists(config.OUTPUT_FOLDER):
    os.makedirs(config.OUTPUT_FOLDER)

def allowed_file(filename):
    if not filename:
        return False
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS

@app.route('/')
def index():
    tasks = session.get('tasks', [])
    return render_template('index.html', has_tasks=len(tasks) > 0)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('Файл не найден', 'error')
        return redirect(url_for('index'))

    uploaded_files = request.files.getlist('file')
    if not uploaded_files or (len(uploaded_files) == 1 and uploaded_files[0].filename == ''):
        flash('Файлы не выбраны', 'error')
        return redirect(url_for('index'))

    model_name = request.form.get('model_name', 'paddle-default')
    if model_name == 'paddle-default' and not ocr_client.is_available():
        flash('Сервис OCR недоступен. Пожалуйста, попробуйте позже.', 'error')
        logger.error("PaddleOCR API is not available")
        return redirect(url_for('index'))

    if 'tasks' not in session:
        session['tasks'] = []
    
    processed_any = False
    detect_seal = request.form.get('detect_seal') == 'on'
    model_name = request.form.get('model_name', 'paddle-default')
    
    for file in uploaded_files:
        if not file.filename or not allowed_file(file.filename):
            continue

        filename = secure_filename(file.filename)
        
        if filename.lower().endswith('.zip'):
            try:
                zip_data = BytesIO(file.read())
                with zipfile.ZipFile(zip_data) as z:
                    for zinfo in z.infolist():
                        if zinfo.is_dir(): continue
                        z_filename = os.path.basename(zinfo.filename)
                        if not z_filename or z_filename.startswith('.') or not allowed_file(z_filename) or z_filename.lower().endswith('.zip'):
                            continue
                        
                        with z.open(zinfo) as zf:
                            file_content = zf.read()
                            try:
                                if model_name == 'paddle-default':
                                    job_data = ocr_client.submit_job(z_filename, file_content, None, detect_seal)
                                else:
                                    job_data = vllm_processor.submit_job(z_filename, file_content, model_name)
                                session['tasks'].append({'task_id': job_data['job_id'], 'filename': z_filename})
                                processed_any = True
                            except Exception as e:
                                logger.error(f"Error sending file {z_filename} from ZIP: {e}")
                                flash(f"Ошибка обработки файла {z_filename} в ZIP архиве: {e}", 'error')
                                continue
            except Exception as e:
                logger.error(f"Error processing ZIP file: {e}")
                flash(f"Ошибка при обработке ZIP архива {filename}: {e}", 'error')
        else:
            try:
                file_content = file.read()
                if model_name == 'paddle-default':
                    job_data = ocr_client.submit_job(filename, file_content, file.content_type, detect_seal)
                else:
                    job_data = vllm_processor.submit_job(filename, file_content, model_name)
                session['tasks'].append({'task_id': job_data['job_id'], 'filename': filename})
                processed_any = True
            except Exception as e:
                logger.error(f"Error sending file to OCR backend: {e}")
                flash(f"Ошибка подключения к OCR бэкенду для {filename}: {e}", 'error')
                continue

    session.modified = True
    if not processed_any:
        if not get_flashed_messages():
            flash('Не удалось обработать ни один файл', 'error')
        return redirect(url_for('index'))
    
    return redirect(url_for('status_page'))

@app.route('/status')
def status_page():
    tasks = session.get('tasks', [])
    if not tasks: return redirect(url_for('index'))
    return render_template('status.html', tasks=tasks)

@app.route('/api/task_status/<task_id>')
def task_status(task_id):
    try:
        processor = get_processor(task_id)
        job_data = processor.get_status(task_id)
        status_map = {'queued': 'processing', 'processing': 'processing', 'completed': 'completed', 'failed': 'failed', 'cancelled': 'failed'}
        status = status_map.get(job_data['status'], 'processing')
        
        progress = 0
        if job_data['total_pages'] > 0:
            progress = int((job_data['processed_pages'] / job_data['total_pages']) * 100)
        elif job_data['status'] == 'completed':
            progress = 100
            
        result = {"status": status, "progress": progress, "step": "ocr" if status == "processing" else "", "total_pages": job_data['total_pages'], "processed_pages": job_data['processed_pages']}
        
        if status == 'completed':
            result["redirect"] = url_for('success', task_id=task_id)
            if 'created_at' in job_data and 'updated_at' in job_data:
                session[f'processing_time_{task_id}'] = round(job_data['updated_at'] - job_data['created_at'], 2)
        
        if status == 'failed':
            result["error"] = job_data.get('error', 'Неизвестная ошибка')
            result["redirect"] = url_for('index')
            
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error checking status: {e}")
        return jsonify({"status": "failed", "error": str(e), "redirect": url_for('index')})

@app.route('/success/<task_id>')
def success(task_id):
    tasks = session.get('tasks', [])
    filename = next((t['filename'] for t in tasks if t['task_id'] == task_id), 'документ')
    processing_time = session.get(f'processing_time_{task_id}', 'Н/Д')
    
    try:
        processor = get_processor(task_id)
        job_data = processor.get_status(task_id)
        detect_seal_enabled = bool(job_data.get('detect_seal', 0))
        result_data = processor.get_result(task_id)
        
        # Auto-save results
        with open(os.path.join(config.OUTPUT_FOLDER, f"{task_id}.json"), "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        
        full_md = "".join([f"# Страница {p['page_num']}\n\n{p['markdown']}\n\n---\n\n" for p in result_data.get('pages', [])])
        with open(os.path.join(config.OUTPUT_FOLDER, f"{task_id}.md"), "w", encoding="utf-8") as f:
            f.write(full_md)
    except Exception as e:
        logger.error(f"Failed to auto-save results: {e}")
        detect_seal_enabled = False

    return render_template('success.html', task_id=task_id, filename=filename, processing_time=processing_time, detect_seal_enabled=detect_seal_enabled)

@app.route('/api/image/<task_id>/<int:page_num>')
def get_image(task_id, page_num):
    try:
        processor = get_processor(task_id)
        if processor == vllm_processor:
            img_data = processor.get_image(task_id, page_num)
            if img_data:
                return send_file(img_data, mimetype='image/png')
            else:
                return "Изображение не найдено", 404
        
        response = ocr_client.get_image(task_id, page_num)
        response.raise_for_status()
        return send_file(response.raw, mimetype='image/png')
    except Exception as e:
        logger.error(f"Error fetching image: {e}")
        return "Изображение не найдено", 404

@app.route('/api/seals/<task_id>')
def list_seals(task_id):
    try:
        processor = get_processor(task_id)
        if processor == vllm_processor:
            return jsonify({"seals": []})
        return jsonify(ocr_client.list_seals(task_id))
    except Exception as e:
        logger.error(f"Error listing seals: {e}")
        return jsonify({"seals": []})

@app.route('/api/seal/<task_id>/<filename>')
def get_seal(task_id, filename):
    try:
        processor = get_processor(task_id)
        if processor == vllm_processor:
            return "Печать не найдена", 404
        response = ocr_client.get_seal(task_id, filename)
        response.raise_for_status()
        return send_file(response.raw, mimetype='image/png')
    except Exception as e:
        logger.error(f"Error fetching seal: {e}")
        return "Печать не найдена", 404

@app.route('/download_result/<task_id>')
def download_result(task_id):
    fmt = request.args.get('format', 'md').lower()
    try:
        processor = get_processor(task_id)
        result_data = processor.get_result(task_id)
        pages = result_data.get('pages', [])
        
        if fmt == 'docx':
            return send_file(converters.to_docx(pages), mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document', as_attachment=True, download_name=f"ocr_result_{task_id}.docx")
        elif fmt == 'xlsx':
            return send_file(converters.to_xlsx(pages), mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', as_attachment=True, download_name=f"ocr_result_{task_id}.xlsx")
        elif fmt == 'extracted':
            return send_file(converters.to_extracted_data_xlsx(pages), mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', as_attachment=True, download_name=f"extracted_data_{task_id}.xlsx")
        else:
            return send_file(converters.to_markdown(pages), mimetype='text/markdown', as_attachment=True, download_name=f"ocr_result_{task_id}.md")
    except Exception as e:
        logger.error(f"Error downloading result: {e}")
        flash(f"Ошибка при скачивании результата: {e}", 'error')
        return redirect(url_for('index'))

@app.route('/api/extracted_data/<task_id>')
def get_extracted_data(task_id):
    try:
        processor = get_processor(task_id)
        result_data = processor.get_result(task_id)
        pages = result_data.get('pages', [])
        extracted = converters.get_extracted_data(pages)
        return jsonify(extracted)
    except Exception as e:
        logger.error(f"Error getting extracted data: {e}")
        return jsonify({"error": str(e)})

@app.route('/new_conversion')
def new_conversion():
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8011)))
