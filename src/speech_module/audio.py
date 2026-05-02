from __future__ import annotations

from io import BytesIO
import queue
import sys

import numpy as np


SAMPLE_RATE = 16000


def load_audio_bytes(file_bytes: bytes) -> tuple[np.ndarray, int]:
    if not file_bytes:
        raise ValueError("Uploaded audio file is empty.")

    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("soundfile is required to read uploaded audio files.") from exc

    try:
        audio_data, sample_rate = sf.read(BytesIO(file_bytes), dtype="float32")
    except Exception as exc:
        raise ValueError("Uploaded file is not a supported audio file.") from exc

    return to_mono_float32(audio_data), int(sample_rate)


def to_mono_float32(audio_data) -> np.ndarray:
    audio_data = np.asarray(audio_data, dtype="float32")
    if audio_data.ndim == 2:
        audio_data = audio_data.mean(axis=1)
    return audio_data.flatten()


def clean_audio(audio_data, sample_rate: int) -> tuple[np.ndarray, int]:
    audio_data = to_mono_float32(audio_data)
    if audio_data.size == 0:
        return audio_data, sample_rate

    try:
        import noisereduce as nr
    except ImportError as exc:
        raise RuntimeError("noisereduce is required to clean audio.") from exc

    reduced_noise = nr.reduce_noise(
        y=audio_data,
        sr=sample_rate,
        stationary=True,
        prop_decrease=0.6,
    )
    max_value = np.max(np.abs(reduced_noise))
    if max_value > 0:
        reduced_noise = reduced_noise / max_value * 0.95

    return reduced_noise.astype("float32"), sample_rate


def record_audio(
    trigger_event=None,
    mouth_active_event=None,
    sample_rate: int = SAMPLE_RATE,
    silence_seconds: float = 3.0,
    max_seconds: float = 60.0,
) -> tuple[np.ndarray, int]:
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError("sounddevice is required for microphone recording.") from exc

    if trigger_event is not None:
        print("Waiting for voice detection from camera...")
        trigger_event.wait()
        trigger_event.clear()
    else:
        input("Press ENTER to start recording...")

    print("Recording. Speak now.")
    audio_queue = queue.Queue()

    def callback(indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        audio_queue.put(indata.copy())

    silence_threshold = 0.005
    max_silence_frames = int(sample_rate * silence_seconds)
    max_total_frames = int(sample_rate * max_seconds)
    recorded_frames = []
    silence_frames = 0
    total_frames = 0

    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        callback=callback,
    ):
        while True:
            data = audio_queue.get()
            recorded_frames.append(data)
            total_frames += len(data)

            if np.max(np.abs(data)) < silence_threshold:
                silence_frames += len(data)
            else:
                silence_frames = 0

            dynamic_max_silence = max_silence_frames
            if mouth_active_event is not None and not mouth_active_event.is_set():
                dynamic_max_silence = int(sample_rate * 0.8)

            if silence_frames > dynamic_max_silence:
                print("Silence detected. Recording finished.")
                break

            if total_frames >= max_total_frames:
                print("Max duration reached. Recording finished.")
                break

    if not recorded_frames:
        return np.zeros((0,), dtype="float32"), sample_rate

    audio_data = np.concatenate(recorded_frames, axis=0).flatten()
    pad_frames = int(sample_rate * 0.3)
    trim_index = max(0, len(audio_data) - silence_frames + pad_frames)
    return audio_data[:trim_index].astype("float32"), sample_rate


def record_audio_until_enter(
    sample_rate: int = SAMPLE_RATE,
) -> tuple[np.ndarray, int]:
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError("sounddevice is required for microphone recording.") from exc

    input("Press ENTER to start recording...")
    print("Recording. Press ENTER again to stop.")

    recorded_frames = []

    def callback(indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        recorded_frames.append(indata.copy())

    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        callback=callback,
    ):
        input("Press ENTER to stop recording...")

    if not recorded_frames:
        return np.zeros((0,), dtype="float32"), sample_rate

    audio_data = np.concatenate(recorded_frames, axis=0).flatten()
    return audio_data.astype("float32"), sample_rate
