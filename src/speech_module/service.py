from __future__ import annotations

import numpy as np

from .audio import clean_audio
from .dom import TranscriptionResult
from .text_processing import clean_text


WHISPER_MODEL_NAME = "tiny"
_whisper_model = None


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        try:
            import whisper
        except ImportError as exc:
            raise RuntimeError("openai-whisper is required for transcription.") from exc

        _whisper_model = whisper.load_model(WHISPER_MODEL_NAME)
    return _whisper_model


def speech_to_text(audio_data, sample_rate: int, language: str = "en") -> str:
    audio_data = np.asarray(audio_data, dtype="float32").flatten()
    if audio_data.size == 0:
        raise ValueError("No audio to transcribe.")

    model = get_whisper_model()
    result = model.transcribe(audio_data, language=language, fp16=False)
    text = str(result.get("text", "")).strip()
    if not text:
        raise ValueError("Could not understand audio.")

    return text


def transcribe_audio(
    audio_data,
    sample_rate: int,
    language: str = "en",
    reduce_noise: bool = True,
) -> TranscriptionResult:
    if reduce_noise:
        audio_data, sample_rate = clean_audio(audio_data, sample_rate)

    raw_text = speech_to_text(audio_data, sample_rate, language=language)
    return TranscriptionResult(
        raw_text=raw_text,
        language=language,
        sentences=clean_text(raw_text),
    )
