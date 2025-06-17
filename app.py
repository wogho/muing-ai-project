import os
import shutil
import uuid
import subprocess
import json # ✨ JSON 처리를 위해 추가된 부분입니다.
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER_BASE = 'static/results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER_BASE'] = RESULT_FOLDER_BASE

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER_BASE, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_audio():
    if 'audio_file' not in request.files: return jsonify({'error': '파일이 없습니다.'}), 400
    file = request.files['audio_file']
    if file.filename == '': return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        task_id = str(uuid.uuid4())
        task_output_dir = os.path.join(app.config['RESULT_FOLDER_BASE'], task_id)
        os.makedirs(task_output_dir)

        command = [
            'python', 'main_pipeline.py',
            filepath,
            '-o', task_output_dir,
            '-s', 'instrumental_mix'
        ]
        subprocess.Popen(command)

        print(f">>> Task {task_id} 시작됨. 클라이언트에 즉시 응답합니다.")
        return jsonify({'status': 'processing', 'task_id': task_id})

    return jsonify({'error': '파일 처리 중 오류 발생'}), 500

@app.route('/status/<task_id>')
def task_status(task_id):
    status_file = os.path.join(app.config['RESULT_FOLDER_BASE'], task_id, 'status.json')
    if os.path.exists(status_file):
        with open(status_file, 'r') as f:
            # 'json'을 사용하기 위해 맨 위에 import가 필요했습니다.
            data = json.load(f)
        return jsonify(data)
    else:
        return jsonify({'status': 'processing'})

@app.route('/static/results/<task_id>/<path:filename>')
def serve_result_file(task_id, filename):
    return send_from_directory(os.path.join(app.config['RESULT_FOLDER_BASE'], task_id), filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
