from pydantic import BaseModel


class SentenceResult(BaseModel):
    sentence: str
    is_question: bool


class TranscriptionResult(BaseModel):
    raw_text: str
    language: str
    sentences: list[SentenceResult]
