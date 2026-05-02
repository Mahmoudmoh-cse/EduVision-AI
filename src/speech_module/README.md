# Speech Module

## Overview

The `speech_module` provides speech-to-text transcription for EduVision. It uses Whisper for transcription, cleans uploaded or recorded audio, and splits the transcript into sentences marked as questions or statements.

## Files

- `api.py`: FastAPI app with health and transcription endpoints
- `dom.py`: Pydantic response models
- `audio.py`: Audio upload loading, noise reduction, and microphone recording
- `service.py`: Lazy Whisper loading and transcription service
- `text_processing.py`: Sentence splitting and question detection
- `Speech_recognition.py`: Small local push-to-talk CLI wrapper

## Run The API

From the project root:

```bash
uvicorn speech_module.api:app --app-dir src --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Transcribe an audio file:

```bash
curl -X POST http://127.0.0.1:8000/transcribe-file \
  -F "file=@question.wav"
```

Example response:

```json
{
  "raw_text": "What is computer vision?",
  "language": "en",
  "sentences": [
    {
      "sentence": "What is computer vision?",
      "is_question": true
    }
  ]
}
```

## Run Local Push-To-Talk

From the project root:

```bash
python src/speech_module/Speech_recognition.py
```

## Dependencies

Install the root project requirements:

```bash
python -m pip install -r requirements.txt
```

Whisper also requires `ffmpeg` to be installed on the system.
