from __future__ import annotations

import json
import sys
import time
from pathlib import Path


try:
    from .audio import clean_audio, record_audio
    from .service import get_whisper_model, speech_to_text, transcribe_audio
    from .text_processing import clean_text
except ImportError:
    SRC_ROOT = Path(__file__).resolve().parents[1]
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))

    from speech_module.audio import clean_audio, record_audio
    from speech_module.service import get_whisper_model, speech_to_text, transcribe_audio
    from speech_module.text_processing import clean_text


def push_to_talk_pipeline(trigger_event=None, mouth_active_event=None):
    start_time = time.perf_counter()
    raw_audio, sample_rate = record_audio(trigger_event, mouth_active_event)
    result = transcribe_audio(raw_audio, sample_rate)

    print(json.dumps(result.model_dump(), indent=2))
    print(f"Total run time: {time.perf_counter() - start_time:.2f} seconds")
    return [sentence.model_dump() for sentence in result.sentences]


def main() -> int:
    print("\n--- Smart Classroom Speech Module ---\n")

    while True:
        try:
            push_to_talk_pipeline()
        except Exception as exc:
            print(f"Speech pipeline failed: {exc}")

        choice = input("Press Q to quit or ENTER to record again: ")
        if choice.lower() == "q":
            print("Exiting system...")
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
