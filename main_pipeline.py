import os
import subprocess
import csv
from collections import defaultdict
import numpy as np
import librosa
import soundfile as sf
import pandas as pd
import shutil
import json
from basic_pitch.inference import predict_and_save
from basic_pitch import ICASSP_2022_MODEL_PATH

# --- 1. 기본 설정 ---
INPUT_AUDIO_FILE = 'my_song.mp3'
OUTPUT_DIR = 'output'
CHORD_ANALYSIS_SOURCE = 'instrumental_mix' # 'instrumental_mix', 'full_mix', 'other', 'bass', 'drums' 중에서 선택

# --- (안정성 강화) 파이프라인 시작 전 폴더 초기화 ---
if os.path.exists(OUTPUT_DIR):
    print(f">>> 기존 '{OUTPUT_DIR}' 폴더의 내용을 모두 삭제합니다.")
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)

# --- 파일 경로 미리 정의 ---
demucs_output_dir = os.path.join(OUTPUT_DIR, 'htdemucs', os.path.splitext(INPUT_AUDIO_FILE)[0])
VOCALS_PATH = os.path.join(demucs_output_dir, 'vocals.wav')
BASS_PATH = os.path.join(demucs_output_dir, 'bass.wav')
DRUMS_PATH = os.path.join(demucs_output_dir, 'drums.wav')
OTHER_PATH = os.path.join(demucs_output_dir, 'other.wav')
MELODY_MIDI_PATH = os.path.join(OUTPUT_DIR, 'melody_output.mid')
MELODY_NOTES_CSV_PATH = os.path.join(OUTPUT_DIR, 'melody_notes.csv')
CHORDS_CSV_PATH = os.path.join(OUTPUT_DIR, 'chords_timeline.csv')
RHYTHM_JSON_PATH = os.path.join(OUTPUT_DIR, 'rhythm_info.json')
TEMP_CHORD_AUDIO_PATH = os.path.join(OUTPUT_DIR, 'temp_chord_audio.wav')

# --- 2. Demucs 스템 분리 ---
print("\n>>> 1. 스템 분리를 시작합니다.")
command = f"demucs \"{INPUT_AUDIO_FILE}\" -o {OUTPUT_DIR}"
subprocess.run(command, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(">>> 스템 분리가 완료되었습니다.")

# --- 3. Basic-Pitch 멜로디 추출 ---
print("\n>>> 2. 멜로디 추출을 시작합니다.")
predict_and_save(
    audio_path_list=[VOCALS_PATH],
    model_or_model_path=ICASSP_2022_MODEL_PATH, output_directory=OUTPUT_DIR,
    save_midi=True, save_notes=True, sonify_midi=False, save_model_outputs=False
)
generated_midi = os.path.join(OUTPUT_DIR, os.path.splitext(os.path.basename(VOCALS_PATH))[0] + '_basic_pitch.mid')
generated_csv = os.path.join(OUTPUT_DIR, os.path.splitext(os.path.basename(VOCALS_PATH))[0] + '_basic_pitch.csv')
os.rename(generated_midi, MELODY_MIDI_PATH)
os.rename(generated_csv, MELODY_NOTES_CSV_PATH)
print(f">>> 멜로디 MIDI 및 노트 데이터 파일 생성이 완료되었습니다.")

# --- 4. 코드 분석용 오디오 소스 준비 ---
print("\n>>> 3. 코드 분석을 위한 오디오 소스를 준비합니다.")
chord_audio_path = ""
if CHORD_ANALYSIS_SOURCE == 'instrumental_mix':
    print(">>> 분석 대상: 반주 전체 (bass + drums + other)")
    bass, sr = librosa.load(BASS_PATH, sr=None)
    drums, _ = librosa.load(DRUMS_PATH, sr=sr)
    other, _ = librosa.load(OTHER_PATH, sr=sr)
    instrumental_mix = bass + drums + other
    sf.write(TEMP_CHORD_AUDIO_PATH, instrumental_mix, sr)
    chord_audio_path = TEMP_CHORD_AUDIO_PATH
elif CHORD_ANALYSIS_SOURCE == 'full_mix':
    print(">>> 분석 대상: 원곡 전체")
    chord_audio_path = INPUT_AUDIO_FILE
else:
    print(f">>> 분석 대상: {CHORD_ANALYSIS_SOURCE} 스템")
    chord_audio_path = os.path.join(demucs_output_dir, f'{CHORD_ANALYSIS_SOURCE}.wav')
    if not os.path.exists(chord_audio_path):
        raise FileNotFoundError(f"{chord_audio_path} 파일이 없습니다.")

# --- 5. Basic-Pitch 반주 노트 데이터 추출 ---
print("\n>>> 4. 코드 분석을 위한 노트 데이터 추출을 시작합니다.")
CHORD_NOTES_CSV_PATH_TEMP = os.path.join(OUTPUT_DIR, os.path.splitext(os.path.basename(chord_audio_path))[0] + '_basic_pitch.csv')
predict_and_save(
    audio_path_list=[chord_audio_path],
    model_or_model_path=ICASSP_2022_MODEL_PATH, output_directory=OUTPUT_DIR,
    save_midi=False, save_notes=True, sonify_midi=False, save_model_outputs=False
)
print(f">>> 코드 분석용 노트 데이터 생성 완료: '{CHORD_NOTES_CSV_PATH_TEMP}'")

# --- 6. 코드(화음) 추정 및 저장 ---
print("\n>>> 5. 코드(화음) 분석 및 데이터 저장을 시작합니다.")
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
CHORD_TEMPLATES = {
    'maj'  : {0, 4, 7}, 'min'  : {0, 3, 7}, 'maj7' : {0, 4, 7, 11},
    'min7' : {0, 3, 7, 10}, 'dom7' : {0, 4, 7, 10}
}
notes_in_time = defaultdict(list)
with open(CHORD_NOTES_CSV_PATH_TEMP, 'r') as f:
    reader = csv.reader(f)
    next(reader) 
    for row in reader:
        notes_in_time[int(float(row[0])/0.5)*0.5].append(int(row[2]))
chord_progression = []
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
    if best_match_score >= 3 and best_match_chord != last_chord:
         chord_progression.append({'time': f"{time_key:.2f}", 'chord': best_match_chord})
         last_chord = best_match_chord
chord_df = pd.DataFrame(chord_progression)
chord_df.to_csv(CHORDS_CSV_PATH, index=False)
print(">>> 코드 진행 데이터를 파일로 저장했습니다.")

# --- 7. 리듬 분석 및 저장 ---
print("\n>>> 6. 리듬(BPM 및 박자) 분석을 시작합니다.")
y_rhythm, sr_rhythm = librosa.load(INPUT_AUDIO_FILE, sr=None)
tempo, beat_frames = librosa.beat.beat_track(y=y_rhythm, sr=sr_rhythm)
beat_times = librosa.frames_to_time(beat_frames, sr=sr_rhythm)
rhythm_data = {
    'bpm': round(float(tempo), 2),
    'beat_times': [round(t, 2) for t in beat_times]
}
with open(RHYTHM_JSON_PATH, 'w') as f:
    json.dump(rhythm_data, f, indent=2)
print(">>> 리듬 데이터를 파일로 저장했습니다.")

# --- 최종 결과물 정리 ---
print("\n--- 최종 분석 결과물 ---")
print(f"✅ 멜로디 MIDI 파일: '{MELODY_MIDI_PATH}'")
print(f"✅ 멜로디 노트 데이터: '{MELODY_NOTES_CSV_PATH}'")
print(f"✅ 코드 진행 데이터: '{CHORDS_CSV_PATH}'")
print(f"✅ 리듬 정보 데이터: '{RHYTHM_JSON_PATH}'")
print("\n모든 파이프라인이 성공적으로 완료되었습니다!")

# --- 임시 파일 삭제 ---
if os.path.exists(TEMP_CHORD_AUDIO_PATH): os.remove(TEMP_CHORD_AUDIO_PATH)
if os.path.exists(CHORD_NOTES_CSV_PATH_TEMP): os.remove(CHORD_NOTES_CSV_PATH_TEMP)
