import argparse
import sys

import speech_recognition as sr
try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*args, **kwargs):
        return None

from speech_capture import capture_speech
from text_cleaner import clean_text


def _check_microphone() -> bool:
    """Verifies that PyAudio and at least one microphone are available."""
    try:
        import pyaudio  # noqa: F401
        mics = sr.Microphone.list_microphone_names()
        if not mics:
            print(" No microphone found. Please connect a microphone.")
            return False
        return True
    except ImportError:
        print(" PyAudio is not installed. Follow these instructions:\n")
        print("  Windows: pip install pyaudio")
        print("           or if that fails:\n           pip install pipwin && pipwin install pyaudio\n")
        print("  macOS  : brew install portaudio && pip install pyaudio")
        print("  Linux  : sudo apt-get install python3-pyaudio portaudio19-dev\n           pip install pyaudio\n")
        return False
    except OSError as e:
        print(f" Microphone error: {e}")
        return False


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Speech-to-Text module for AI Educational Assistant"
    )
    parser.add_argument(
        "--lang",
        default="en-US",
        help="BCP-47 language tag for speech recognition (default: en-US)",
    )
    args = parser.parse_args()
    lang: str = args.lang

    if not _check_microphone():
        sys.exit(1)

    print("Microphone ready")

    try:
        while True:
            print("\n Speak now... (Ctrl+C to exit)")

            try:
                raw_text = capture_speech(lang)
                print(f"[RAW]     {raw_text}")

                cleaned_text = clean_text(raw_text, lang)
                print(f"[CLEANED] {cleaned_text}")

            except sr.UnknownValueError:
                print("Could not understand audio, please try again")
            except sr.RequestError as e:
                print(f"Google API error: {e}")
            except Exception as e:
                print(f"Error: {e}")

            print("─" * 50)

    except KeyboardInterrupt:
        print("\n Exiting")
        sys.exit(0)


if __name__ == "__main__":
    main()
