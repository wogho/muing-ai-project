import os
import json
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from main_pipeline import run_full_pipeline # <-- 우리의 분석 엔진 함수를 import 합니다.

app = Flask(__name__)

# 파일 업로드 및 결과물 저장을 위한 폴더 설정
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    """메인 페이지를 렌더링합니다."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_route():
    """오디오 파일을 받아 저장하고, 분석 파이프라인을 실행합니다."""
    if 'audio_file' not in request.files:
        return jsonify({'error': '오디오 파일이 없습니다.'}), 400
    
    file = request.files['audio_file']
    if file.filename == '':
        return jsonify({'error': '선택된 파일이 없습니다.'}), 400

    if file:
        filename = secure_filename(file.filename)
        upload_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_filepath)
        
        try:
            # AI 분석 엔진 호출
            print(f"AI 분석 시작: {upload_filepath}")
            timeline_json_path = run_full_pipeline(upload_filepath, OUTPUT_FOLDER, 'instrumental_mix')
            
            # 생성된 JSON 파일의 내용을 읽음
            with open(timeline_json_path, 'r', encoding='utf-8') as f:
                timeline_data = json.load(f)
            
            print("AI 분석 완료. 결과를 반환합니다.")
            return jsonify(timeline_data)

        except Exception as e:
            # 오류 발생 시 자세한 내용을 로그로 남김
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'분석 중 오류가 발생했습니다: {str(e)}'}), 500
    
    return jsonify({'error': '알 수 없는 오류가 발생했습니다.'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)

