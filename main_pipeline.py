import os
import subprocess
import csv
from collections import defaultdict
import numpy as np
import librosa
import soundfile as sf
from basic_pitch.inference import predict_and_save
from basic_pitch import ICASSP_2022_MODEL_PATH

# --- 1. 기본 설정 ---
INPUT_AUDIO_FILE = 'my_song.mp3'
OUTPUT_DIR = 'output'

# ################################################################## #
# #####        (사용자 제안 적용) 코드 분석 소스 선택        ##### #
# ################################################################## #
# 'instrumental_mix', 'full_mix', 'other', 'bass', 'drums' 중에서 선택
CHORD_ANALYSIS_SOURCE = 'instrumental_mix' 

# --- 파일 경로 미리 정의 ---
VOCALS_PATH = os.path.join(OUTPUT_DIR, 'htdemucs', os.path.splitext(INPUT_AUDIO_FILE)[0], 'vocals.wav')
BASS_PATH = os.path.join(OUTPUT_DIR, 'htdemucs', os.path.splitext(INPUT_AUDIO_FILE)[0], 'bass.wav')
DRUMS_PATH = os.path.join(OUTPUT_DIR, 'htdemucs', os.path.splitext(INPUT_AUDIO_FILE)[0], 'drums.wav')
OTHER_PATH = os.path.join(OUTPUT_DIR, 'htdemucs', os.path.splitext(INPUT_AUDIO_FILE)[0], 'other.wav')
MELODY_MIDI_PATH = os.path.join(OUTPUT_DIR, 'melody_output.mid')
TEMP_CHORD_AUDIO_PATH = os.path.join(OUTPUT_DIR, 'temp_chord_audio.wav') # 임시 분석 파일

# --- 2. Demucs 스템 분리 ---
print(">>> 1. 스템 분리를 시작합니다.")
if not os.path.exists(VOCALS_PATH):
    command = f"demucs {INPUT_AUDIO_FILE} -o {OUTPUT_DIR}"
    subprocess.run(command, shell=True, check=True)
    print(">>> 스템 분리가 완료되었습니다.")
else:
    print(">>> 기존 분리된 파일을 사용합니다. (건너뛰기)")

# --- 3. Basic-Pitch 멜로디 추출 ---
# (이전과 동일)

# --- 4. 코드 분석을 위한 오디오 소스 준비 ---
print("\n>>> 3. 코드 분석을 위한 오디오 소스를 준비합니다.")

if CHORD_ANALYSIS_SOURCE == 'instrumental_mix':
    print(">>> 분석 대상: 반주 전체 (bass + drums + other)")
    # 각 스템 로드
    bass, sr = librosa.load(BASS_PATH, sr=None)
    drums, _ = librosa.load(DRUMS_PATH, sr=sr)
    other, _ = librosa.load(OTHER_PATH, sr=sr)
    # 모든 반주 스템을 합침
    instrumental_mix = bass + drums + other
    # 임시 파일로 저장
    sf.write(TEMP_CHORD_AUDIO_PATH, instrumental_mix, sr)
    chord_audio_path = TEMP_CHORD_AUDIO_PATH

elif CHORD_ANALYSIS_SOURCE == 'full_mix':
    print(">>> 분석 대상: 원곡 전체")
    chord_audio_path = INPUT_AUDIO_FILE
else:
    print(f">>> 분석 대상: {CHORD_ANALYSIS_SOURCE} 스템")
    chord_audio_path = os.path.join(OUTPUT_DIR, 'htdemucs', os.path.splitext(INPUT_AUDIO_FILE)[0], f'{CHORD_ANALYSIS_SOURCE}.wav')

if not os.path.exists(chord_audio_path) and CHORD_ANALYSIS_SOURCE != 'full_mix':
     raise FileNotFoundError(f"{chord_audio_path} 파일이 없습니다. 스템 분리가 올바르게 되었는지 확인해주세요.")

# --- 5. Basic-Pitch 노트 데이터 추출 ---
CHORD_NOTES_CSV_PATH = os.path.join(OUTPUT_DIR, os.path.splitext(os.path.basename(chord_audio_path))[0] + '_basic_pitch.csv')
print("\n>>> 4. 코드 분석을 위한 노트 데이터 추출을 시작합니다.")
# (이하 로직은 이전과 거의 동일)
if not os.path.exists(CHORD_NOTES_CSV_PATH):
    predict_and_save(
        audio_path_list=[chord_audio_path],
        model_or_model_path=ICASSP_2022_MODEL_PATH, output_directory=OUTPUT_DIR,
        save_midi=False, sonify_midi=False, save_model_outputs=False, save_notes=True
    )
    print(f">>> 코드 분석용 노트 데이터 생성 완료: '{CHORD_NOTES_CSV_PATH}'")
else:
    print(f">>> 기존 생성된 코드 분석용 노트 데이터를 사용합니다. (건너뛰기)")

# --- 6. 노트 데이터를 기반으로 코드(화음) 추정 ---
print("\n>>> 5. 코드(화음) 분석을 시작합니다.")
# (이하 코드 추정 로직은 이전과 동일)
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
CHORD_TEMPLATES = { 'maj': {0, 4, 7}, 'min': {0, 3, 7} }
notes_in_time = defaultdict(list)
with open(CHORD_NOTES_CSV_PATH, 'r') as f:
    reader = csv.reader(f)
    next(reader) 
    for row in reader:
        start_time, midi_note = float(row[0]), int(row[2])
        time_key = int(start_time / 0.5) * 0.5
        notes_in_time[time_key].append(midi_note)
print("\n--- 코드 진행 결과 ---")
last_chord = None
for time_key in sorted(notes_in_time.keys()):
    pitch_classes = {note % 12 for note in notes_in_time[time_key]}
    best_match_chord, best_match_score = 'N', 0
    for root in range(12):
        for chord_type, template in CHORD_TEMPLATES.items():
            chord_notes = {(root + i) % 12 for i in template}
            score = len(pitch_classes.intersection(chord_notes))
            if score > best_match_score:
                best_match_score, best_match_chord = score, f"{NOTE_NAMES[root]}:{chord_type}"
    if best_match_score >= 2 and best_match_chord != last_chord:
         print(f"시간: {time_key:05.2f}초  |  추정 코드: {best_match_chord}")
         last_chord = best_match_chord
print("\n✅ 모든 파이프라인이 성공적으로 완료되었습니다!")

# 임시 파일 삭제
if os.path.exists(TEMP_CHORD_AUDIO_PATH):
    os.remove(TEMP_CHORD_AUDIO_PATH)