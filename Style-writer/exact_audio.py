from glob import glob
import subprocess
import os
import pdb

video_dir = "../../dataset/videos"
audio_dir = "../../dataset/audios"
video_files = glob(f"{video_dir}/*.mp4")
print(video_files)

for video_path in video_files:
    audo_path = os.path.join(audio_dir ,os.path.splitext(os.path.basename(video_path))[0].split("/")[-1] + '.mp3')
    print(audo_path)
    ffmpeg_command = [
        "ffmpeg",
        "-i", video_path, 
        "-f", "mp3",
        "-y",
        "-vn", audo_path
    ]
    # 调用ffmpeg命令
    try:
        subprocess.run(ffmpeg_command, check=True)
        print("Conversion completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

