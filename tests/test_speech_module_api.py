import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from speech_module import api as api_module
from speech_module.dom import TranscriptionResult


def test_health_returns_ok():
    client = TestClient(api_module.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_transcribe_file_returns_mocked_result(monkeypatch):
    def fake_load_audio_bytes(file_bytes):
        assert file_bytes == b"fake audio"
        return np.array([0.1, 0.2], dtype="float32"), 16000

    def fake_transcribe_audio(audio_data, sample_rate, language="en"):
        assert sample_rate == 16000
        assert language == "en"
        return TranscriptionResult(
            raw_text="What is computer vision?",
            language=language,
            sentences=[
                {
                    "sentence": "What is computer vision?",
                    "is_question": True,
                }
            ],
        )

    monkeypatch.setattr(api_module, "load_audio_bytes", fake_load_audio_bytes)
    monkeypatch.setattr(api_module, "transcribe_audio", fake_transcribe_audio)
    client = TestClient(api_module.app)

    response = client.post(
        "/transcribe-file",
        content=b"fake audio",
        headers={"content-type": "audio/wav"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "raw_text": "What is computer vision?",
        "language": "en",
        "sentences": [
            {
                "sentence": "What is computer vision?",
                "is_question": True,
            }
        ],
    }


def test_transcribe_file_accepts_multipart_upload(monkeypatch):
    pytest.importorskip("multipart")

    def fake_load_audio_bytes(file_bytes):
        assert file_bytes == b"fake audio"
        return np.array([0.1, 0.2], dtype="float32"), 16000

    def fake_transcribe_audio(audio_data, sample_rate, language="en"):
        return TranscriptionResult(
            raw_text="How does Whisper work?",
            language=language,
            sentences=[
                {
                    "sentence": "How does Whisper work?",
                    "is_question": True,
                }
            ],
        )

    monkeypatch.setattr(api_module, "load_audio_bytes", fake_load_audio_bytes)
    monkeypatch.setattr(api_module, "transcribe_audio", fake_transcribe_audio)
    client = TestClient(api_module.app)

    response = client.post(
        "/transcribe-file",
        files={"file": ("question.wav", b"fake audio", "audio/wav")},
    )

    assert response.status_code == 200
    assert response.json()["raw_text"] == "How does Whisper work?"


def test_transcribe_file_rejects_missing_audio():
    client = TestClient(api_module.app)

    response = client.post(
        "/transcribe-file",
        content=b"",
        headers={"content-type": "audio/wav"},
    )

    assert response.status_code == 422


def test_transcribe_file_rejects_invalid_audio(monkeypatch):
    def fake_load_audio_bytes(file_bytes):
        raise ValueError("Uploaded file is not a supported audio file.")

    monkeypatch.setattr(api_module, "load_audio_bytes", fake_load_audio_bytes)
    client = TestClient(api_module.app)

    response = client.post(
        "/transcribe-file",
        content=b"not audio",
        headers={"content-type": "audio/wav"},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "Uploaded file is not a supported audio file."
