import os
import subprocess
from basic_pitch.inference import predict_and_save
from basic_pitch import ICASSP_2022_MODEL_PATH

# --- 1. 기본 설정 ---
# 분석하고 싶은 오디오 파일을 여기에 지정하세요.
INPUT_AUDIO_FILE = 'my_song.mp3'
# 결과물이 저장될 폴더 이름입니다.
OUTPUT_DIR = 'output'

# Demucs가 생성할 보컬 파일의 전체 경로를 미리 정의합니다.
VOCALS_PATH = os.path.join(OUTPUT_DIR, 'htdemucs', os.path.splitext(INPUT_AUDIO_FILE)[0], 'vocals.wav')
# 최종적으로 생성될 MIDI 파일의 전체 경로입니다.
MIDI_OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'melody_output.mid')

print(">>> 1. 스템 분리를 시작합니다. (시간이 다소 소요될 수 있습니다)")

# --- 2. Demucs를 이용한 스템 분리 ---
# 이전에 이미 분리한 파일이 없다면, demucs 명령어를 실행합니다.
if not os.path.exists(VOCALS_PATH):
    # demucs 명령어를 실행하여 4개 스템(보컬, 드럼, 베이스, 그 외)으로 분리합니다.
    command = f"demucs {INPUT_AUDIO_FILE} -o {OUTPUT_DIR}"
    # 터미널에서 명령어를 실행합니다.
    subprocess.run(command, shell=True, check=True)
    print(f">>> '{VOCALS_PATH}' 파일이 생성되었습니다.")
else:
    print(f">>> 기존에 분리된 '{VOCALS_PATH}' 파일을 사용합니다. (건너뛰기)")

print("\n>>> 2. 멜로디 추출 및 MIDI 생성을 시작합니다.")

# --- 3. Basic-Pitch를 이용한 멜로디 추출 및 MIDI 변환 ---
# Basic-Pitch는 오디오 파일 경로만 알려주면, 멜로디 분석부터 MIDI 변환/저장까지 한 번에 처리해줍니다.
predict_and_save(
    audio_path_list=[VOCALS_PATH],             # 분석할 오디오 파일(보컬) 목록
    model_or_model_path=ICASSP_2022_MODEL_PATH,# 사용할 AI 모델
    output_directory=OUTPUT_DIR,               # 결과물을 저장할 폴더
    save_midi=True,                            # MIDI 파일 저장 활성화
    sonify_midi=False,                         # MIDI를 소리 파일로 만드는 기능 비활성화
    save_model_outputs=False,                  # 중간 분석 데이터 저장 비활성화
    save_notes=False                           # 노트 데이터를 CSV로 저장하는 기능 비활성화
)

# Basic-Pitch가 생성한 MIDI 파일의 이름은 'vocals_basic_pitch.mid'와 같은 형식입니다.
# 이 파일의 이름을 우리가 원하는 'melody_output.mid'로 변경합니다.
generated_midi_name = os.path.splitext(os.path.basename(VOCALS_PATH))[0] + '_basic_pitch.mid'
generated_midi_path = os.path.join(OUTPUT_DIR, generated_midi_name)
os.rename(generated_midi_path, MIDI_OUTPUT_PATH)

print(f"\n✅ 파이프라인 완료! '{MIDI_OUTPUT_PATH}' 파일이 생성되었습니다.")
print("파일을 다운로드해서 확인해보세요!")
