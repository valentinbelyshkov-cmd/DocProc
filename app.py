"""
Flask application for PDF OCR Converter.
Main entry point for the web application.
"""
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
from processors.vllm_processor import VLLMProcessor
from processors.classic_processor import ClassicProcessor
import converters

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(config)
app.secret_key = config.SECRET_KEY

vllm_processor = VLLMProcessor()
classic_processor = ClassicProcessor()


def get_processor(task_id):
    """Get the processor that owns the given task."""
    if task_id in vllm_processor.tasks:
        return vllm_processor
    if task_id in classic_processor.tasks:
        return classic_processor
    return None


if not os.path.exists(config.UPLOAD_FOLDER):
    os.makedirs(config.UPLOAD_FOLDER)

if not os.path.exists(config.OUTPUT_FOLDER):
    os.makedirs(config.OUTPUT_FOLDER)

if not os.path.exists(config.DEBUG_IMAGES_FOLDER):
    os.makedirs(config.DEBUG_IMAGES_FOLDER)


def allowed_file(filename):
    """Check if file extension is allowed."""
    if not filename:
        return False
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS


@app.route('/api/health')
def api_health():
    """API endpoint for checking service status."""
    try:
        return jsonify({
            "status": "ok",
            "vllm_processor": True,
            "classic_processor": True
        })
    except Exception as e:
        logger.error(f"Error checking status: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        })


@app.route('/api/document_types')
def api_document_types():
    """Get list of supported document types."""
    from processors.handlers_registry import DocumentHandlerRegistry

    doc_types = []
    for doc_type in DocumentHandlerRegistry.get_document_types():
        info = DocumentHandlerRegistry.get_handler_info(doc_type)
        if info:
            doc_types.append(info)

    return jsonify({"document_types": doc_types})


@app.route('/api/models')
def api_models():
    """Get list of available OCR models."""
    from models.models_registry import ModelRegistry

    models = []
    for model_name in ModelRegistry.get_available_models():
        info = ModelRegistry.get_model_info(model_name)
        if info:
            models.append(info)

    return jsonify({"models": models})


@app.route('/')
def index():
    """Main page."""
    tasks = session.get('tasks', [])
    return render_template('index.html', has_tasks=len(tasks) > 0)


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and start OCR processing."""
    if 'file' not in request.files:
        flash('Файл не найден', 'error')
        return redirect(url_for('index'))

    uploaded_files = request.files.getlist('file')
    if not uploaded_files or (len(uploaded_files) == 1 and uploaded_files[0].filename == ''):
        flash('Файлы не выбраны', 'error')
        return redirect(url_for('index'))

    model_name = request.form.get('model_name', 'glm-ocr')
    detect_seal = request.form.get('detect_seal') == 'on'
    
    logger.info(f"Upload received: model_name={model_name}, detect_seal={detect_seal}")

    if 'tasks' not in session:
        session['tasks'] = []

    processed_any = False

    for file in uploaded_files:
        if not file.filename or not allowed_file(file.filename):
            logger.warning(f"File skipped (invalid or no name): {file.filename}")
            continue

        filename = secure_filename(file.filename)
        
        # Read content to log its size
        file_content = file.read()
        content_length = len(file_content)
        logger.info(f"Processing uploaded file: {filename}, size: {content_length} bytes")

        if filename.lower().endswith('.zip'):
            try:
                zip_data = BytesIO(file_content)
                with zipfile.ZipFile(zip_data) as z:
                    for zinfo in z.infolist():
                        if zinfo.is_dir():
                            continue
                        z_filename = os.path.basename(zinfo.filename)
                        if not z_filename or z_filename.startswith('.') or not allowed_file(z_filename) or z_filename.lower().endswith('.zip'):
                            continue

                        with z.open(zinfo) as zf:
                            file_content = zf.read()
                            try:
                                if model_name in ['tesseract', 'easyocr', 'pyocr']:
                                    job_data = classic_processor.submit_job(z_filename, file_content, model_name)
                                else:
                                    job_data = vllm_processor.submit_job(z_filename, file_content, model_name, detect_seal=detect_seal)
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
                if model_name in ['tesseract', 'easyocr', 'pyocr']:
                    job_data = classic_processor.submit_job(filename, file_content, model_name)
                else:
                    job_data = vllm_processor.submit_job(filename, file_content, model_name, detect_seal=detect_seal)
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
    """Show processing status page."""
    tasks = session.get('tasks', [])
    if not tasks:
        return redirect(url_for('index'))
    return render_template('status.html', tasks=tasks)


@app.route('/api/task_status/<task_id>')
def task_status(task_id):
    """Get task status for polling."""
    try:
        processor = get_processor(task_id)
        if processor is None:
            return jsonify({"status": "failed", "error": "Task not found", "redirect": url_for('index')})
        
        job_data = processor.get_status(task_id)

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
            if 'created_at' in job_data and 'updated_at' in job_data:
                session[f'processing_time_{task_id}'] = round(job_data['updated_at'] - job_data['created_at'], 2)

        if status == 'failed':
            error_msg = job_data.get('error', 'Неизвестная ошибка')
            result["error"] = error_msg
            result["redirect"] = url_for('index')

        return jsonify(result)
    except Exception as e:
        logger.error(f"Error checking status: {e}")
        return jsonify({"status": "failed", "error": str(e), "redirect": url_for('index')})


@app.route('/success/<task_id>')
def success(task_id):
    """Show success page with results."""
    tasks = session.get('tasks', [])
    filename = next((t['filename'] for t in tasks if t['task_id'] == task_id), 'документ')
    processing_time = session.get(f'processing_time_{task_id}', 'Н/Д')

    try:
        processor = get_processor(task_id)
        if processor:
            job_data = processor.get_status(task_id)
            detect_seal_enabled = bool(job_data.get('detect_seal', 0))
            result_data = processor.get_result(task_id)

            with open(os.path.join(config.OUTPUT_FOLDER, f"{task_id}.json"), "w", encoding="utf-8") as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)

            full_md = "".join([f"# Страница {p['page_num']}\n\n{p['markdown']}\n\n---\n\n" for p in result_data.get('pages', [])])
            with open(os.path.join(config.OUTPUT_FOLDER, f"{task_id}.md"), "w", encoding="utf-8") as f:
                f.write(full_md)
        else:
            detect_seal_enabled = False
    except Exception as e:
        logger.error(f"Failed to auto-save results: {e}")
        detect_seal_enabled = False

    return render_template('success.html', task_id=task_id, filename=filename, processing_time=processing_time, detect_seal_enabled=detect_seal_enabled)


@app.route('/api/image/<task_id>/<int:page_num>')
def get_image(task_id, page_num):
    """Get page image."""
    try:
        processor = get_processor(task_id)
        if processor:
            img_data = processor.get_image(task_id, page_num)
            if img_data:
                return send_file(BytesIO(img_data), mimetype='image/png')
        return "Изображение не найдено", 404
    except Exception as e:
        logger.error(f"Error fetching image: {e}")
        return "Изображение не найдено", 404


@app.route('/api/seals/<task_id>')
def list_seals(task_id):
    """List detected seals."""
    try:
        processor = get_processor(task_id)
        if processor:
            return jsonify({"seals": []})
        return jsonify({"seals": []})
    except Exception as e:
        logger.error(f"Error listing seals: {e}")
        return jsonify({"seals": []})


@app.route('/api/seal/<task_id>/<filename>')
def get_seal(task_id, filename):
    """Get seal image."""
    return "Печать не найдена", 404


@app.route('/download_result/<task_id>')
def download_result(task_id):
    """Download OCR result in specified format."""
    fmt = request.args.get('format', 'md').lower()
    try:
        processor = get_processor(task_id)
        if processor is None:
            flash('Результат не найден', 'error')
            return redirect(url_for('index'))
            
        result_data = processor.get_result(task_id)
        pages = result_data.get('pages', [])

        if fmt == 'docx':
            return send_file(
                converters.to_docx(pages),
                mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                as_attachment=True,
                download_name=f"ocr_result_{task_id}.docx"
            )
        elif fmt == 'xlsx':
            return send_file(
                converters.to_xlsx(pages),
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                as_attachment=True,
                download_name=f"ocr_result_{task_id}.xlsx"
            )
        elif fmt == 'extracted':
            return send_file(
                converters.to_extracted_data_xlsx(pages),
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                as_attachment=True,
                download_name=f"extracted_data_{task_id}.xlsx"
            )
        else:
            return send_file(
                converters.to_markdown(pages),
                mimetype='text/markdown',
                as_attachment=True,
                download_name=f"ocr_result_{task_id}.md"
            )
    except Exception as e:
        logger.error(f"Error downloading result: {e}")
        flash(f"Ошибка при скачивании результата: {e}", 'error')
        return redirect(url_for('index'))


@app.route('/api/extracted_data/<task_id>')
def get_extracted_data(task_id):
    """Get extracted document data as JSON."""
    try:
        processor = get_processor(task_id)
        if processor is None:
            return jsonify({"error": "Task not found"})
            
        result_data = processor.get_result(task_id)
        pages = result_data.get('pages', [])
        extracted = converters.get_extracted_data(pages)
        return jsonify(extracted)
    except Exception as e:
        logger.error(f"Error getting extracted data: {e}")
        return jsonify({"error": str(e)})


@app.route('/new_conversion')
def new_conversion():
    """Start new conversion (clear session)."""
    session.clear()
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8011)))