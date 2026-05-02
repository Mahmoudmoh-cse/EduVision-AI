import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from main import run_pipeline
from speech_module.dom import TranscriptionResult


def test_question_sentence_is_classified():
    output = []
    classified_questions = []

    def fake_record():
        return [0.1, 0.2], 16000

    def fake_transcribe(raw_audio, sample_rate):
        return TranscriptionResult(
            raw_text="What is binary search?",
            language="en",
            sentences=[
                {
                    "sentence": "What is binary search?",
                    "is_question": True,
                }
            ],
        )

    def fake_classify(question):
        classified_questions.append(question)
        return "Algorithms", 0.98

    results = run_pipeline(
        record_fn=fake_record,
        transcribe_fn=fake_transcribe,
        classify_fn=fake_classify,
        printer=output.append,
    )

    assert classified_questions == ["What is binary search?"]
    assert results[0].predicted_topic == "Algorithms"
    assert results[0].score == 0.98
    assert any("Subject: Algorithms" in line for line in output)


def test_statement_only_transcript_skips_classification():
    output = []

    def fake_record():
        return [0.1], 16000

    def fake_transcribe(raw_audio, sample_rate):
        return TranscriptionResult(
            raw_text="Today we are learning about arrays.",
            language="en",
            sentences=[
                {
                    "sentence": "Today we are learning about arrays.",
                    "is_question": False,
                }
            ],
        )

    def fail_if_called(question):
        raise AssertionError("Statements should not be sent to NLP classification")

    results = run_pipeline(
        record_fn=fake_record,
        transcribe_fn=fake_transcribe,
        classify_fn=fail_if_called,
        printer=output.append,
    )

    assert results == []
    assert any("No question detected" in line for line in output)


def test_multiple_questions_are_all_classified():
    classified_questions = []

    def fake_record():
        return [0.1], 16000

    def fake_transcribe(raw_audio, sample_rate):
        return TranscriptionResult(
            raw_text="What is a CNN? How does tokenization work?",
            language="en",
            sentences=[
                {
                    "sentence": "What is a CNN?",
                    "is_question": True,
                },
                {
                    "sentence": "How does tokenization work?",
                    "is_question": True,
                },
            ],
        )

    def fake_classify(question):
        classified_questions.append(question)
        if "CNN" in question:
            return "Computer Vision", 0.91
        return "Natural Language Processing", 0.89

    results = run_pipeline(
        record_fn=fake_record,
        transcribe_fn=fake_transcribe,
        classify_fn=fake_classify,
        printer=lambda message: None,
    )

    assert classified_questions == [
        "What is a CNN?",
        "How does tokenization work?",
    ]
    assert [result.predicted_topic for result in results] == [
        "Computer Vision",
        "Natural Language Processing",
    ]
