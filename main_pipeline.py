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

def run_full_pipeline(input_file, output_dir, chord_source):
    # --- 1. 폴더 초기화 ---
    print(f">>> 0. 출력 폴더 '{output_dir}'를 초기화합니다.")
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # --- 2. 파일 경로 정의 ---
    demucs_output_dir = os.path.join(output_dir, 'htdemucs', os.path.splitext(os.path.basename(input_file))[0])
    vocals_path = os.path.join(demucs_output_dir, 'vocals.wav'); bass_path = os.path.join(demucs_output_dir, 'bass.wav'); drums_path = os.path.join(demucs_output_dir, 'drums.wav'); other_path = os.path.join(demucs_output_dir, 'other.wav')
    melody_midi_path = os.path.join(output_dir, 'melody_output.mid'); melody_notes_csv_path = os.path.join(output_dir, 'melody_notes.csv')
    chords_csv_path = os.path.join(output_dir, 'chords_timeline.csv'); rhythm_json_path = os.path.join(output_dir, 'rhythm_info.json')
    timeline_json_path = os.path.join(output_dir, 'muing_timeline.json'); temp_chord_audio_path = os.path.join(output_dir, 'temp_chord_audio.wav')

    # --- 3. 스템 분리 ---
    print("\n>>> 1. 스템 분리를 시작합니다.")
    command = f"demucs \"{input_file}\" -o {output_dir}"
    try:
        subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(">>> 스템 분리가 완료되었습니다.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Demucs 실행 중 오류가 발생했습니다.\n--- STDERR ---\n{e.stderr}"); raise e

    # --- 4. 멜로디 추출 ---
    print("\n>>> 2. 멜로디 추출을 시작합니다.")
    predict_and_save(
        audio_path_list=[vocals_path], model_or_model_path=ICASSP_2022_MODEL_PATH, output_directory=output_dir,
        save_midi=True, save_notes=True, sonify_midi=False, save_model_outputs=False
    )
    generated_midi = os.path.join(output_dir, os.path.splitext(os.path.basename(vocals_path))[0] + '_basic_pitch.mid')
    generated_csv = os.path.join(output_dir, os.path.splitext(os.path.basename(vocals_path))[0] + '_basic_pitch.csv')
    os.rename(generated_midi, melody_midi_path); os.rename(generated_csv, melody_notes_csv_path)
    print(f">>> 멜로디 MIDI 및 노트 데이터 파일 생성이 완료되었습니다.")

    # --- 5. 코드 분석 ---
    print("\n>>> 3. 코드 분석을 시작합니다.")
    if chord_source == 'instrumental_mix':
        print(">>> 분석 대상: 반주 전체 (bass + drums + other)")
        bass, sr = librosa.load(bass_path, sr=None); drums, _ = librosa.load(drums_path, sr=sr); other, _ = librosa.load(other_path, sr=sr)
        instrumental_mix = bass + drums + other; sf.write(temp_chord_audio_path, instrumental_mix, sr); chord_audio_path = temp_chord_audio_path
    else:
        chord_audio_path = input_file if chord_source == 'full_mix' else os.path.join(demucs_output_dir, f'{chord_source}.wav')
    
    temp_notes_csv_path = os.path.join(output_dir, os.path.splitext(os.path.basename(chord_audio_path))[0] + '_basic_pitch.csv')
    predict_and_save(
        audio_path_list=[chord_audio_path], model_or_model_path=ICASSP_2022_MODEL_PATH, output_directory=output_dir,
        save_midi=False, save_notes=True, sonify_midi=False, save_model_outputs=False
    )
    NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']; CHORD_TEMPLATES = {'maj':{0,4,7}, 'min':{0,3,7}, 'dom7':{0,4,7,10}}
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
                chord_notes = {(root+i)%12 for i in template}; score = len(pitch_classes.intersection(chord_notes))
                if score > best_match_score: best_match_score, best_match_chord = score,f"{NOTE_NAMES[root]}:{chord_type}"
        if best_match_score >= 2 and best_match_chord != last_chord: chord_progression.append({'time':f"{time_key:.2f}",'chord':best_match_chord}); last_chord = best_match_chord
    pd.DataFrame(chord_progression).to_csv(chords_csv_path, index=False)
    print(">>> 코드 진행 데이터를 파일로 저장했습니다.")

    # --- 6. 리듬 분석 ---
    print("\n>>> 6. 리듬(BPM 및 박자) 분석을 시작합니다.")
    y, sr = librosa.load(input_file, sr=None); tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr)
    rhythm_data = {'bpm': round(float(tempo), 2), 'beat_times': [round(t, 2) for t in beat_times]}
    with open(rhythm_json_path, 'w') as f: json.dump(rhythm_data, f, indent=2)
    print(">>> 리듬 데이터를 파일로 저장했습니다.")

    # --- 7. 데이터 통합 ---
    print("\n>>> 7. 모든 분석 데이터를 하나의 타임라인으로 통합합니다.")
    all_events = []
    with open(melody_notes_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # --- ✨ 요청하신 수정사항 + 디버그 코드 모두 적용 ✨ ---
            
            # 1. 디버깅을 위해 현재 처리 중인 row의 내용을 출력합니다.
            print(f"DEBUG: 현재 row의 내용: {row}") 

            # 2. 'velocity_midi' 키가 없을 경우를 대비해 기본값(64)을 사용하여 오류를 방지합니다.
            velocity = int(row.get('velocity_midi', 64))
            
            # 3. 타임라인 이벤트 리스트에 노트를 추가합니다.
            all_events.append({
                'time': round(float(row['start_time_s']), 2),
                'type': 'note',
                'pitch': int(row['pitch_midi']),
                'duration': round(float(row['end_time_s']) - float(row['start_time_s']), 2),
                'velocity': velocity
            })
            # --- ✨ 적용 끝 ✨ ---
            
    with open(chords_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader: all_events.append({'time':round(float(row['time']),2),'type':'chord','value':row['chord']})
    for beat_time in rhythm_data['beat_times']: all_events.append({'time':beat_time,'type':'beat'})
    all_events.sort(key=lambda x: x['time'])
    final_timeline_data = { 'bpm': rhythm_data['bpm'], 'events': all_events }
    with open(timeline_json_path, 'w') as f: json.dump(final_timeline_data, f, indent=2, ensure_ascii=False)
    print(f">>> 최종 통합 데이터 파일을 생성했습니다: '{timeline_json_path}'")

    # --- 8. 임시 파일 삭제 ---
    if os.path.exists(temp_chord_audio_path): os.remove(temp_chord_audio_path)
    if os.path.exists(temp_notes_csv_path): os.remove(temp_notes_csv_path)
    
    return timeline_json_path

def main():
    parser = argparse.ArgumentParser(description="Muing AI: 오디오 파일을 분석하여 멜로디, 코드, 리듬을 추출합니다.")
    parser.add_argument("input_file", type=str, help="분석할 오디오 파일의 경로")
    parser.add_argument("-o", "--output_dir", type=str, default="output", help="결과물이 저장될 폴더")
    parser.add_argument("-s", "--source", type=str, default="instrumental_mix", help="코드 분석에 사용할 소스")
    args = parser.parse_args()
    timeline_json_path = run_full_pipeline(args.input_file, args.output_dir, args.source)
    print(f"\n👑 최종 통합 타임라인: '{timeline_json_path}'"); print("\n모든 파이프라인이 성공적으로 완료되었습니다!")

if __name__ == "__main__":
    main()
