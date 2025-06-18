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

def find_stem_file(directory, stem_name):
    """'stem.wav' 또는 'stem.mp3' 파일을 찾아 경로를 반환합니다."""
    wav_path = os.path.join(directory, f'{stem_name}.wav')
    mp3_path = os.path.join(directory, f'{stem_name}.mp3')
    
    if os.path.exists(wav_path):
        return wav_path
    elif os.path.exists(mp3_path):
        return mp3_path
    else:
        return None

def initialize_directory(output_dir):
    """파이프라인 시작 전, 출력 폴더를 초기화합니다."""
    print(f">>> 0. 출력 폴더 '{output_dir}'를 초기화합니다.")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

def run_stem_separation(input_file, output_dir):
    """Demucs를 실행하여 오디오 파일에서 스템을 분리합니다."""
    print("\n>>> 1. 스템 분리를 시작합니다.")
    # --mp3 옵션을 추가하여 출력을 mp3로 유도해볼 수 있습니다.
    command = f"demucs -j 1 --shifts 0 --segment 7 --mp3 \"{input_file}\" -o \"{output_dir}\""
    try:
        subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(">>> 스템 분리가 완료되었습니다.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Demucs 실행 중 오류가 발생했습니다.\n--- STDERR ---\n{e.stderr}")
        raise e

def extract_melody(input_file, output_dir):
    """분리된 보컬 트랙에서 멜로디를 추출합니다."""
    print("\n>>> 2. 멜로디 추출을 시작합니다.")
    demucs_output_dir = os.path.join(output_dir, 'htdemucs', os.path.splitext(os.path.basename(input_file))[0])
    
    vocals_path = find_stem_file(demucs_output_dir, 'vocals')
    if not vocals_path:
        raise FileNotFoundError("보컬 파일(vocals.wav 또는 vocals.mp3)을 찾을 수 없습니다.")

    predict_and_save(
        audio_path_list=[vocals_path],
        model_or_model_path=ICASSP_2022_MODEL_PATH,
        output_directory=output_dir,
        save_midi=True, save_notes=True, sonify_midi=False, save_model_outputs=False
    )
    generated_midi = os.path.join(output_dir, os.path.splitext(os.path.basename(vocals_path))[0] + '_basic_pitch.mid')
    generated_csv = os.path.join(output_dir, os.path.splitext(os.path.basename(vocals_path))[0] + '_basic_pitch.csv')
    os.rename(generated_midi, os.path.join(output_dir, 'melody_output.mid'))
    os.rename(generated_csv, os.path.join(output_dir, 'melody_notes.csv'))
    print(f">>> 멜로디 MIDI 및 노트 데이터 파일 생성이 완료되었습니다.")
    return os.path.join(output_dir, 'melody_notes.csv'), os.path.join(output_dir, 'melody_output.mid')

def extract_chords(input_file, output_dir, source):
    """설정된 소스에서 코드를 분석하고 CSV 파일로 저장합니다."""
    print("\n>>> 3. 코드 분석을 시작합니다.")
    demucs_output_dir = os.path.join(output_dir, 'htdemucs', os.path.splitext(os.path.basename(input_file))[0])
    temp_chord_audio_path = os.path.join(output_dir, 'temp_chord_audio.wav')
    chords_csv_path = os.path.join(output_dir, 'chords_timeline.csv')
    
    chord_audio_path = ""
    if source == 'instrumental_mix':
        print(">>> 분석 대상: 반주 전체 (bass + drums + other)")
        bass_path = find_stem_file(demucs_output_dir, 'bass')
        drums_path = find_stem_file(demucs_output_dir, 'drums')
        other_path = find_stem_file(demucs_output_dir, 'other')
        if not all([bass_path, drums_path, other_path]):
            raise FileNotFoundError("코드 분석에 필요한 반주 스템 파일 중 일부를 찾을 수 없습니다.")
        
        bass, sr = librosa.load(bass_path, sr=None); drums, _ = librosa.load(drums_path, sr=sr); other, _ = librosa.load(other_path, sr=sr)
        instrumental_mix = bass + drums + other
        sf.write(temp_chord_audio_path, instrumental_mix, sr)
        chord_audio_path = temp_chord_audio_path
    elif source == 'full_mix':
        print(">>> 분석 대상: 원곡 전체")
        chord_audio_path = input_file
    else:
        print(f">>> 분석 대상: {source} 스템")
        chord_audio_path = find_stem_file(demucs_output_dir, source)
        if not chord_audio_path: raise FileNotFoundError(f"{source} 스템 파일을 찾을 수 없습니다.")

    print("\n>>> 4. 코드 분석을 위한 노트 데이터 추출을 시작합니다.")
    temp_notes_csv_path = os.path.join(output_dir, os.path.splitext(os.path.basename(chord_audio_path))[0] + '_basic_pitch.csv')
    predict_and_save(
        audio_path_list=[chord_audio_path],
        model_or_model_path=ICASSP_2022_MODEL_PATH,
        output_directory=output_dir,
        save_midi=False, save_notes=True, sonify_midi=False, save_model_outputs=False
    )
    print(f">>> 코드 분석용 노트 데이터 생성 완료: '{temp_notes_csv_path}'")

    print("\n>>> 5. 코드(화음) 분석 및 데이터 저장을 시작합니다.")
    NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    CHORD_TEMPLATES = {'maj':{0,4,7}, 'min':{0,3,7}, 'dom7':{0,4,7,10}, 'maj7':{0,4,7,11}, 'min7':{0,3,7,10}}
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
                if score > best_match_score:
                    best_match_score, best_match_chord = score, f"{NOTE_NAMES[root]}:{chord_type}"
        if best_match_score >= 3 and best_match_chord != last_chord:
             chord_progression.append({'time': f"{time_key:.2f}", 'chord': best_match_chord})
             last_chord = best_match_chord
    pd.DataFrame(chord_progression).to_csv(chords_csv_path, index=False)
    print(">>> 코드 진행 데이터를 파일로 저장했습니다.")
    
    if os.path.exists(temp_chord_audio_path): os.remove(temp_chord_audio_path)
    if os.path.exists(temp_notes_csv_path): os.remove(temp_notes_csv_path)
    
    return chords_csv_path

def extract_rhythm(input_file, output_dir):
    print("\n>>> 6. 리듬(BPM 및 박자) 분석을 시작합니다.")
    rhythm_json_path = os.path.join(output_dir, 'rhythm_info.json')
    y, sr = librosa.load(input_file, sr=None)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr)
    rhythm_data = {'bpm': round(tempo.item(), 2), 'beat_times': [round(t, 2) for t in beat_times]}
    with open(rhythm_json_path, 'w') as f: json.dump(rhythm_data, f, indent=2)
    print(">>> 리듬 데이터를 파일로 저장했습니다.")
    return rhythm_json_path

def integrate_data(output_dir, melody_notes_csv, chords_csv, rhythm_json):
    print("\n>>> 7. 모든 분석 데이터를 하나의 타임라인으로 통합합니다.")
    timeline_json_path = os.path.join(output_dir, 'muing_timeline.json')
    all_events = []
    with open(melody_notes_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader: all_events.append({'time': round(float(row['start_time_s']), 2), 'type': 'note', 'pitch': int(row['pitch_midi']), 'duration': round(float(row['end_time_s']) - float(row['start_time_s']), 2), 'velocity': int(row.get('velocity_midi', 64))})
    with open(chords_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader: all_events.append({'time':round(float(row['time']),2),'type':'chord','value':row['chord']})
    with open(rhythm_json, 'r') as f:
        rhythm_data = json.load(f)
        for beat_time in rhythm_data['beat_times']: all_events.append({'time':beat_time,'type':'beat'})
    all_events.sort(key=lambda x: x['time'])
    final_timeline_data = { 'bpm': rhythm_data['bpm'], 'events': all_events }
    with open(timeline_json_path, 'w') as f: json.dump(final_timeline_data, f, indent=2, ensure_ascii=False)
    print(f">>> 최종 통합 데이터 파일을 생성했습니다: '{timeline_json_path}'")
    return timeline_json_path

def run_full_pipeline(input_file, output_dir, chord_source):
    try:
        initialize_directory(output_dir)
        filename = os.path.basename(input_file)

        # 1. 스템 분리를 먼저 실행합니다.
        run_stem_separation(input_file, output_dir)
        
        # 2. 모든 분석을 먼저 수행합니다 (파일을 옮기기 *전*에).
        melody_notes_csv_path, _ = extract_melody(input_file, output_dir)
        chords_csv_path = extract_chords(input_file, output_dir, chord_source)
        rhythm_json_path = extract_rhythm(input_file, output_dir)
        
        # 3. 모든 분석이 끝난 후, 결과 오디오 파일들을 정리하고 이동시킵니다.
        demucs_output_dir = os.path.join(output_dir, 'htdemucs', os.path.splitext(filename)[0])
        stems = ['vocals', 'drums', 'bass', 'other']
        result_audio_files = {}

        for stem in stems:
            original_path = find_stem_file(demucs_output_dir, stem)
            if original_path:
                actual_filename = os.path.basename(original_path)
                final_path = os.path.join(output_dir, actual_filename)
                shutil.move(original_path, final_path)
                result_audio_files[stem] = actual_filename
            else:
                print(f"경고: {stem} 스템 파일을 찾을 수 없습니다.")
        
        # 4. 분석에 사용된 빈 Demucs 폴더 구조를 삭제합니다.
        if os.path.exists(os.path.join(output_dir, 'htdemucs')):
             shutil.rmtree(os.path.join(output_dir, 'htdemucs'))

        # 5. 모든 데이터를 최종 타임라인으로 통합합니다.
        timeline_json_path = integrate_data(output_dir, melody_notes_csv_path, chords_csv_path, rhythm_json_path)

        # 6. 성공 상태를 status.json에 기록합니다.
        status_data = {
            'status': 'complete',
            'message': '모든 분석이 완료되었습니다.',
            'original_filename': filename,
            'result_files': {stem: f"/static/results/{os.path.basename(output_dir)}/{fname}" for stem, fname in result_audio_files.items()},
            'timeline_file': f"/static/results/{os.path.basename(output_dir)}/muing_timeline.json"
        }
        with open(os.path.join(output_dir, 'status.json'), 'w') as f:
            json.dump(status_data, f)
        
        print(">>> Status.json 파일 생성 완료. 모든 작업이 끝났습니다.")

    except Exception as e:
        status_data = {'status': 'error', 'message': str(e)}
        with open(os.path.join(output_dir, 'status.json'), 'w') as f:
            json.dump(status_data, f)
        print(f"❌ 파이프라인 실행 중 오류 발생: {e}")

def main():
    """명령어 라인에서 직접 실행될 때 사용되는 함수입니다."""
    parser = argparse.ArgumentParser(description="Muing AI: 오디오 파일을 분석하여 멜로디, 코드, 리듬을 추출합니다.")
    parser.add_argument("input_file", type=str, help="분석할 오디오 파일의 경로")
    parser.add_argument("-o", "--output_dir", type=str, default="output", help="결과물이 저장될 폴더")
    parser.add_argument("-s", "--source", type=str, default="instrumental_mix", 
                        choices=['instrumental_mix','full_mix','other','bass','drums'],
                        help="코드 분석에 사용할 소스")
    args = parser.parse_args()

    run_full_pipeline(args.input_file, args.output_dir, args.source)
    
    print("\n모든 파이프라인이 성공적으로 완료되었습니다!")

if __name__ == "__main__":
    main()