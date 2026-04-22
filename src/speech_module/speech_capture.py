# -*- coding: utf-8 -*-

import speech_recognition as sr


def capture_speech(lang: str = "en-US") -> str:
    """
    Captures audio from the microphone and transcribes it using Google Speech-to-Text.

    Args:
        lang: BCP-47 language tag (e.g. 'en-US', 'ar-EG').

    Returns:
        Raw transcribed string.

    Raises:
        sr.UnknownValueError: If speech could not be recognized.
        sr.RequestError: If the Google Speech API request failed.
    """
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=15)

    raw_text = recognizer.recognize_google(audio, language=lang)
    return raw_text
