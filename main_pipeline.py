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
import argparse
from basic_pitch.inference import predict_and_save
from basic_pitch import ICASSP_2022_MODEL_PATH

def initialize_directory(output_dir):
    """파이프라인 시작 전, 출력 폴더를 초기화합니다."""
    print(f">>> 0. 출력 폴더 '{output_dir}'를 초기화합니다.")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

def run_stem_separation(input_file, output_dir):
    """Demucs를 실행하여 오디오 파일에서 스템을 분리합니다."""
    print("\n>>> 1. 스템 분리를 시작합니다.")
    command = f"demucs \"{input_file}\" -o {output_dir}"
    subprocess.run(command, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(">>> 스템 분리가 완료되었습니다.")

def extract_melody(input_file, output_dir):
    """분리된 보컬 트랙에서 멜로디를 추출하여 MIDI 및 노트 CSV 파일로 저장합니다."""
    print("\n>>> 2. 멜로디 추출을 시작합니다.")
    demucs_output_dir = os.path.join(output_dir, 'htdemucs', os.path.splitext(os.path.basename(input_file))[0])
    vocals_path = os.path.join(demucs_output_dir, 'vocals.wav')
    melody_midi_path = os.path.join(output_dir, 'melody_output.mid')
    melody_notes_csv_path = os.path.join(output_dir, 'melody_notes.csv')

    predict_and_save(
        audio_path_list=[vocals_path],
        model_or_model_path=ICASSP_2022_MODEL_PATH, output_directory=output_dir,
        save_midi=True, save_notes=True, sonify_midi=False, save_model_outputs=False
    )
    generated_midi = os.path.join(output_dir, os.path.splitext(os.path.basename(vocals_path))[0] + '_basic_pitch.mid')
    generated_csv = os.path.join(output_dir, os.path.splitext(os.path.basename(vocals_path))[0] + '_basic_pitch.csv')
    os.rename(generated_midi, melody_midi_path)
    os.rename(generated_csv, melody_notes_csv_path)
    print(f">>> 멜로디 MIDI 및 노트 데이터 파일 생성이 완료되었습니다.")
    return melody_notes_csv_path, melody_midi_path

def extract_chords(input_file, output_dir, source):
    """설정된 소스에서 코드를 분석하고 CSV 파일로 저장합니다."""
    demucs_output_dir = os.path.join(output_dir, 'htdemucs', os.path.splitext(os.path.basename(input_file))[0])
    temp_chord_audio_path = os.path.join(output_dir, 'temp_chord_audio.wav')
    chords_csv_path = os.path.join(output_dir, 'chords_timeline.csv')
    
    # 1. 분석용 오디오 소스 준비
    print("\n>>> 3. 코드 분석을 위한 오디오 소스를 준비합니다.")
    chord_audio_path = ""
    if source == 'instrumental_mix':
        print(">>> 분석 대상: 반주 전체 (bass + drums + other)")
        bass_path = os.path.join(demucs_output_dir, 'bass.wav')
        drums_path = os.path.join(demucs_output_dir, 'drums.wav')
        other_path = os.path.join(demucs_output_dir, 'other.wav')
        bass, sr = librosa.load(bass_path, sr=None); drums, _ = librosa.load(drums_path, sr=sr); other, _ = librosa.load(other_path, sr=sr)
        instrumental_mix = bass + drums + other
        sf.write(temp_chord_audio_path, instrumental_mix, sr)
        chord_audio_path = temp_chord_audio_path
    elif source == 'full_mix':
        print(">>> 분석 대상: 원곡 전체")
        chord_audio_path = input_file
    else:
        print(f">>> 분석 대상: {source} 스템")
        chord_audio_path = os.path.join(demucs_output_dir, f'{source}.wav')
        if not os.path.exists(chord_audio_path): raise FileNotFoundError(f"{chord_audio_path} 파일이 없습니다.")

    # 2. Basic-Pitch로 노트 데이터 추출
    print("\n>>> 4. 코드 분석을 위한 노트 데이터 추출을 시작합니다.")
    temp_notes_csv_path = os.path.join(output_dir, os.path.splitext(os.path.basename(chord_audio_path))[0] + '_basic_pitch.csv')
    predict_and_save(
        audio_path_list=[chord_audio_path],
        model_or_model_path=ICASSP_2022_MODEL_PATH, output_directory=output_dir,
        save_midi=False, save_notes=True, sonify_midi=False, save_model_outputs=False
    )
    print(f">>> 코드 분석용 노트 데이터 생성 완료: '{temp_notes_csv_path}'")

    # 3. 노트 데이터로 코드 추정 및 저장
    print("\n>>> 5. 코드(화음) 분석 및 데이터 저장을 시작합니다.")
    NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    CHORD_TEMPLATES = {'maj':{0,4,7},'min':{0,3,7},'maj7':{0,4,7,11},'min7':{0,3,7,10},'dom7':{0,4,7,10}}
    notes_in_time = defaultdict(list)
    with open(temp_notes_csv_path, 'r') as f:
        reader = csv.reader(f); next(reader)
        for row in reader: notes_in_time[int(float(row[0])/0.5)*0.5].append(int(row[2]))
    chord_progression = []
    last_chord = None
    for time_key in sorted(notes_in_time.keys()):
        pitch_classes = {note % 12 for note in notes_in_time[time_key]}
        best_match_chord, best_match_score = 'N', 0
        for root in range(12):
            for chord_type, template in CHORD_TEMPLATES.items():
                chord_notes = {(root + i) % 12 for i in template}
                score = len(pitch_classes.intersection(chord_notes))
                if score > best_match_score: best_match_score, best_match_chord = score, f"{NOTE_NAMES[root]}:{chord_type}"
        if best_match_score >= 3 and best_match_chord != last_chord:
             chord_progression.append({'time': f"{time_key:.2f}", 'chord': best_match_chord})
             last_chord = best_match_chord
    chord_df = pd.DataFrame(chord_progression)
    chord_df.to_csv(chords_csv_path, index=False)
    print(">>> 코드 진행 데이터를 파일로 저장했습니다.")
    if os.path.exists(temp_chord_audio_path): os.remove(temp_chord_audio_path)
    if os.path.exists(temp_notes_csv_path): os.remove(temp_notes_csv_path)
    return chords_csv_path

def extract_rhythm(input_file, output_dir):
    """원곡에서 리듬(BPM, 박자)을 분석하고 JSON 파일로 저장합니다."""
    print("\n>>> 6. 리듬(BPM 및 박자) 분석을 시작합니다.")
    rhythm_json_path = os.path.join(output_dir, 'rhythm_info.json')
    y, sr = librosa.load(input_file, sr=None)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr)
    rhythm_data = {'bpm': round(float(tempo), 2), 'beat_times': [round(t, 2) for t in beat_times]}
    with open(rhythm_json_path, 'w') as f: json.dump(rhythm_data, f, indent=2)
    print(">>> 리듬 데이터를 파일로 저장했습니다.")
    return rhythm_json_path

def integrate_data(output_dir, melody_notes_csv, chords_csv, rhythm_json):
    """생성된 모든 분석 데이터를 하나의 타임라인 JSON 파일로 통합합니다."""
    print("\n>>> 7. 모든 분석 데이터를 하나의 타임라인으로 통합합니다.")
    timeline_json_path = os.path.join(output_dir, 'muing_timeline.json')
    # ... (데이터 통합 로직) ...
    print(f">>> 최종 통합 데이터 파일을 생성했습니다: '{timeline_json_path}'")
    return timeline_json_path

def main():
    """메인 파이프라인을 실행하는 함수"""
    parser = argparse.ArgumentParser(description="Muing AI: 오디오 파일을 분석하여 멜로디, 코드, 리듬을 추출합니다.")
    parser.add_argument("input_file", type=str, help="분석할 오디오 파일의 경로 (예: my_song.mp3)")
    parser.add_argument("-o", "--output_dir", type=str, default="output", help="결과물이 저장될 폴더 이름 (기본값: output)")
    parser.add_argument("-s", "--source", type=str, default="instrumental_mix", 
                        choices=['instrumental_mix', 'full_mix', 'other', 'bass', 'drums'],
                        help="코드 분석에 사용할 소스 (기본값: instrumental_mix)")
    args = parser.parse_args()

    initialize_directory(args.output_dir)
    run_stem_separation(args.input_file, args.output_dir)
    melody_notes_csv, melody_midi = extract_melody(args.input_file, args.output_dir)
    chords_csv = extract_chords(args.input_file, args.output_dir, args.source)
    rhythm_json = extract_rhythm(args.input_file, args.output_dir)
    timeline_json = integrate_data(args.output_dir, melody_notes_csv, chords_csv, rhythm_json)

    print("\n--- 최종 분석 결과물 ---")
    print(f"✅ 멜로디 MIDI 파일: '{melody_midi}'")
    # ... (기타 최종 결과물 출력)
    print("\n모든 파이프라인이 성공적으로 완료되었습니다!")

if __name__ == "__main__":
    main()
