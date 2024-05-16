
import requests
from glob import glob
import pdb
import os
# Whisper Asr Webservice
url = "http://localhost:9000/asr"

# 设置查询参数
params = {
    "encode": "true",
    "task": "transcribe",
    "word_timestamps": "false",
    "output": "txt",
    "initial_prompt": "输出简体中文。",
    "language": "zh"
}

# 设置请求头
headers = {
    "accept": "application/json",
    # "Content-Type": "multipart/form-data",
}

audio_dir = "../../dataset/audios"
text_dir = "./dataset"
audio_files = glob(f"{audio_dir}/*.mp3")
for audio_path in audio_files:
    files = {
        "audio_file": (os.path.basename(audio_path), open(audio_path, "rb"), "audio/mpeg")
    }

    response = requests.post(url, params=params, headers=headers, files=files)
    output_text_path = os.path.join(text_dir, os.path.splitext(os.path.basename(audio_path))[0] + '.txt' )
    print(output_text_path)
    with open(output_text_path, 'w') as f:
        f.write(response.text)

