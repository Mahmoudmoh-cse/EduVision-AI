
import whisper
import tempfile
import subprocess
import os
from tkinter import Tk, filedialog



# Extract audio
def extract_audio(video_path, audio_path):

    process = subprocess.run([
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        audio_path
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if process.returncode != 0:
        print("❌ FFMPEG ERROR:\n")
        print(process.stderr.decode())
        raise RuntimeError("ffmpeg failed")



# Transcribe video

def transcribe_video(video_path):

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_path = tmp.name

    extract_audio(video_path, audio_path)

    print("Loading Whisper...")
    model = whisper.load_model("base")

    print("Transcribing...")
    result = model.transcribe(audio_path)

    os.remove(audio_path)

    print("\n====================")
    print("TRANSCRIPTION:")
    print("====================\n")
    print(result["text"])
    print("\n====================")

    return result["text"]



# FILE PICKER 

def choose_file():

    Tk().withdraw()  # hide window
    file_path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov")]
    )
    return file_path



# MAIN

if __name__ == "__main__":

    print("📂 Choose a video file...")

    video_path = choose_file()

    if not video_path:
        print(" No file selected!")
    else:
        transcribe_video(video_path)
