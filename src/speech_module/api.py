from fastapi import FastAPI, HTTPException, Request

from .audio import load_audio_bytes
from .dom import TranscriptionResult
from .service import transcribe_audio


app = FastAPI(title="EduVision Speech Module")


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/transcribe-file", response_model=TranscriptionResult)
async def transcribe_file(request: Request, language: str = "en"):
    file_bytes = await _read_audio_payload(request)

    try:
        audio_data, sample_rate = load_audio_bytes(file_bytes)
        return transcribe_audio(audio_data, sample_rate, language=language)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


async def _read_audio_payload(request: Request) -> bytes:
    content_type = request.headers.get("content-type", "")

    if content_type.startswith("multipart/form-data"):
        try:
            form = await request.form()
        except AssertionError as exc:
            raise HTTPException(
                status_code=500,
                detail="python-multipart is required to parse uploaded files.",
            ) from exc

        upload = form.get("file")
        if upload is None or not hasattr(upload, "read"):
            raise HTTPException(status_code=422, detail="Missing uploaded file field 'file'.")

        file_bytes = await upload.read()
    else:
        file_bytes = await request.body()

    if not file_bytes:
        raise HTTPException(status_code=422, detail="Uploaded audio file is empty.")

    return file_bytes
