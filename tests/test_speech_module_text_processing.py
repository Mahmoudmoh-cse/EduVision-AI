import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from speech_module.text_processing import clean_text


def test_clean_text_detects_statement():
    result = clean_text("today we are learning about arrays")

    assert result == [
        {
            "sentence": "Today we are learning about arrays.",
            "is_question": False,
        }
    ]


def test_clean_text_detects_direct_question():
    result = clean_text("what is binary search")

    assert result == [
        {
            "sentence": "What is binary search?",
            "is_question": True,
        }
    ]


def test_clean_text_splits_embedded_question():
    result = clean_text("I understand arrays but what is binary search")

    assert result == [
        {
            "sentence": "I understand arrays.",
            "is_question": False,
        },
        {
            "sentence": "But what is binary search?",
            "is_question": True,
        },
    ]
