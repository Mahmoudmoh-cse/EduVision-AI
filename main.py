from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from speech_module.audio import record_audio_until_enter  # noqa: E402
from speech_module.service import transcribe_audio  # noqa: E402


@dataclass
class SubjectClassification:
    question: str
    predicted_topic: str
    score: float


def classify_subject(question: str) -> tuple[str, float]:
    try:
        from nlp_module.classifier import classify_question
    except Exception as exc:
        raise RuntimeError(
            "NLP classifier could not be imported. Run "
            "`python -m pip install -r requirements.txt` and try again."
        ) from exc

    try:
        return classify_question(question)
    except Exception as exc:
        raise RuntimeError(
            "NLP classification failed while loading or running the transformer model."
        ) from exc


def check_nlp_dependencies() -> None:
    try:
        from nlp_module.classifier import classify_question as _classify_question
    except Exception as exc:
        raise RuntimeError(
            "NLP classifier dependencies are not ready. Run "
            "`python -m pip install -r requirements.txt` before using main.py."
        ) from exc


def run_pipeline(
    record_fn: Callable = record_audio_until_enter,
    transcribe_fn: Callable = transcribe_audio,
    classify_fn: Callable[[str], tuple[str, float]] = classify_subject,
    printer: Callable[[str], None] = print,
) -> list[SubjectClassification]:
    raw_audio, sample_rate = record_fn()
    transcription = transcribe_fn(raw_audio, sample_rate)

    printer("\nTranscript:")
    printer(transcription.raw_text or "(empty)")

    questions = []
    printer("\nDetected sentences:")
    if not transcription.sentences:
        printer("  No sentences detected.")

    for index, sentence in enumerate(transcription.sentences, start=1):
        label = "QUESTION" if sentence.is_question else "Statement"
        printer(f"  [{index}] {label}: {sentence.sentence}")
        if sentence.is_question:
            questions.append(sentence.sentence)

    if not questions:
        printer("\nNo question detected. NLP classification skipped.")
        return []

    results = []
    printer("\nSubject classification:")
    for question in questions:
        predicted_topic, score = classify_fn(question)
        result = SubjectClassification(
            question=question,
            predicted_topic=str(predicted_topic),
            score=float(score),
        )
        results.append(result)
        printer(f"  {result.question}")
        printer(f"    Subject: {result.predicted_topic} (score: {result.score:.2f})")

    return results


def main() -> int:
    print("\n--- EduVision Push-To-Talk Subject Classifier ---\n")

    try:
        check_nlp_dependencies()
    except RuntimeError as exc:
        print(exc)
        return 1

    while True:
        try:
            run_pipeline()
        except KeyboardInterrupt:
            print("\nExiting system...")
            return 130
        except Exception as exc:
            print(f"\nPipeline failed: {exc}")

        choice = input("\nPress Q to quit or ENTER to ask another question: ")
        if choice.lower() == "q":
            print("Exiting system...")
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
