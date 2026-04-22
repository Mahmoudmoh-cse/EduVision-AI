"""
Smart Classroom Speech Module — Bilingual Edition
==================================================
• Arabic speech  → automatically translated to English
• English speech → kept as-is
• All UI messages in English
• Multi-pass noise reduction + intonation heuristic
• Google Speech API (en-US first, then ar-EG fallback)
• Optional OpenAI Whisper with task="translate" (always outputs English)

Dependencies:
    pip install noisereduce librosa soundfile speechrecognition numpy scipy deep-translator

Optional (offline, best accuracy):
    pip install openai-whisper
    # Also requires ffmpeg in PATH: https://ffmpeg.org/download.html
"""

import os
import sys

# ── Windows UTF-8 fix ─────────────────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
# ─────────────────────────────────────────────────────────────────────────────

import re
import json
import numpy as np
import soundfile as sf
import librosa
import noisereduce as nr
import speech_recognition as sr
from scipy.signal import lfilter
from datetime import datetime


# ── CONFIGURATION ─────────────────────────────────────────────────────────────
USE_WHISPER       = False      # True → Whisper (offline, auto-translate, best accuracy)
WHISPER_MODEL     = "medium"   # tiny | base | small | medium | large
PRIMARY_DIALECT   = "ar-EG"    # Arabic dialect for Google fallback
FALLBACK_DIALECTS = ["ar-SA", "ar-LB", "ar"]
OUTPUT_DIR        = os.path.dirname(os.path.abspath(__file__))
# ─────────────────────────────────────────────────────────────────────────────

def _out(name):
    return os.path.join(OUTPUT_DIR, name)


# ── 1. QUESTION DETECTION ENGINE (English) ────────────────────────────────────
# All text is English (Arabic gets translated), so patterns are English-only.

_D2_OPENER = re.compile(
    r"(?:^|\.\s+|\?\s+|!\s+)"
    r"(what|why|how|when|where|who|which|whose|whom"
    r"|can|could|would|should|shall|will|may|might|must"
    r"|is|are|was|were|do|does|did|have|has|had)\b",
    re.IGNORECASE,
)

_D3_EMBEDDED = re.compile(
    r"\b("
    r"do you|does he|does she|does it|do we|do they"
    r"|did you|did he|did she|did it|did we|did they"
    r"|are you|is he|is she|is it|are we|are they"
    r"|was he|was she|was it|were you|were we|were they"
    r"|have you|has he|has she|have we|have they"
    r"|had you|had he|had she|had we|had they"
    r"|can you|can he|can she|can we|can they"
    r"|could you|could he|could she|could we|could they"
    r"|would you|would he|would she|would we|would they"
    r"|should you|should he|should she|should we|should they"
    r"|will you|will he|will she|will we|will they"
    r"|shall we|shall I"
    r"|is there|are there|was there|were there"
    r"|is this|is that|are these|are those"
    r"|what is|what are|what was|what were|what do|what did|what have"
    r"|why is|why are|why do|why did|why would|why can|why should"
    r"|how is|how are|how do|how did|how can|how would|how should"
    r"|how much|how many|how long|how often|how far"
    r"|when is|when are|when do|when did|when will|when would|when can"
    r"|where is|where are|where do|where did|where will|where would|where can"
    r"|who is|who are|who was|who were|who will|who would|who can|who did"
    r"|which is|which are|which was|which do|which did|which would"
    r")\b",
    re.IGNORECASE,
)

_D4_TAG = re.compile(
    r"\b("
    r"right\??|correct\??|yeah\??|ok\??"
    r"|isn'?t it|aren'?t you|wasn'?t it|weren'?t they"
    r"|don'?t you|doesn'?t it|didn'?t you|didn'?t they"
    r"|haven'?t you|hasn'?t it|hadn'?t you"
    r"|can'?t you|couldn'?t you|won'?t you|wouldn'?t you"
    r"|shouldn'?t it|shouldn'?t you|mustn'?t it"
    r"|do you think|don'?t you think|would you say"
    r"|you know\??"
    r")\b",
    re.IGNORECASE,
)

_D5_INDIRECT = re.compile(
    r"\b("
    r"I wonder (if|whether|what|why|how|when|where|who)"
    r"|tell me (what|why|how|when|where|who|if|whether)"
    r"|I('?m| am) not sure (if|whether|what|why|how)"
    r"|do you know (if|whether|what|why|how|when|where|who)"
    r"|I want to know (if|whether|what|why|how)"
    r"|can you (tell|explain|show|describe|clarify)"
    r"|could you (tell|explain|show|describe|clarify)"
    r"|would you (mind|like|be able to)"
    r"|I('?d| would) like to know"
    r"|any idea (if|whether|what|why|how|when|where|who)"
    r")\b",
    re.IGNORECASE,
)

_GREETINGS = re.compile(
    r"^(hello|hi|hey|good morning|good afternoon|good evening|greetings"
    r"|how are you|how('?s| is) it going|how('?s| is) everything)\b",
    re.IGNORECASE,
)


def _is_question(text: str) -> bool:
    s = text.strip()
    if not s:
        return False
    if _GREETINGS.match(s):
        return False
    if "?" in s:
        return True
    if _D2_OPENER.search(s):
        return True
    if _D3_EMBEDDED.search(s):
        return True
    if _D4_TAG.search(s):
        return True
    if _D5_INDIRECT.search(s):
        return True
    return False


# ── SENTENCE SPLITTER (English) ───────────────────────────────────────────────

_SPLIT_MARKERS = re.compile(
    r"\s+(?="
    r"but (what|why|how|when|where|who|can|could|do|does|did|is|are|was|were|will|would|have|has)"
    r"|so (what|why|how|when|where|who|can|could|do|does|did|is|are|will|would)"
    r"|and (what|why|how|when|where|who|can|could|do|does|did|is|are|will|would)"
    r"|now (what|why|how|when|where|who|can|could|do|does|did|is|are|will|would)"
    r"|like (what|why|how|when|where|who)"
    r"|i mean (what|why|how|when|where|who)"
    r")",
    re.IGNORECASE,
)

_QUESTION_STARTERS = re.compile(
    r"\s+(?="
    r"(?:what|why|how|when|where|who|which)"
    r"\s+(?:is|are|was|were|do|does|did|can|could|will|would|should|have|has|had)\b"
    r")",
    re.IGNORECASE,
)


def _split_by_embedded_questions(text: str) -> list:
    parts = _SPLIT_MARKERS.split(text)
    if len(parts) == 1:
        parts = _QUESTION_STARTERS.split(text)
    return [p.strip() for p in parts if p.strip()]


def _split_into_sentences(text: str) -> list:
    text = " ".join(text.split())
    if any(p in text for p in [".", "?", "!"]):
        parts = re.split(r"(?<=[.?!])\s+", text)
        final = []
        for p in parts:
            final.extend(_split_by_embedded_questions(p))
        return [s.strip() for s in final if s.strip()]
    parts = re.split(
        r"\s+(?:and then|but then|however|also|because|after that"
        r"|okay so|alright so|now|anyway)\s+",
        text, flags=re.IGNORECASE,
    )
    final = []
    for p in parts:
        final.extend(_split_by_embedded_questions(p))
    return [s.strip() for s in final if s.strip()]


def _format_sentence(sentence: str, question: bool) -> str:
    s = sentence.strip().rstrip(".?!")
    if not s:
        return s
    s = s[0].upper() + s[1:]
    return s + ("?" if question else ".")


def clean_text(text: str) -> list:
    sentences = _split_into_sentences(text)
    results = []
    for sent in sentences:
        q = _is_question(sent)
        results.append({"sentence": _format_sentence(sent, q), "is_question": q})
    return results


# ── 2. LANGUAGE DETECTION & TRANSLATION ──────────────────────────────────────

def _is_arabic(text: str) -> bool:
    """Returns True if text contains Arabic Unicode characters."""
    return bool(re.search(r"[\u0600-\u06FF]", text))


def _translate_to_english(text: str) -> str:
    """Translate Arabic text to English using deep-translator (free, no API key)."""
    try:
        from deep_translator import GoogleTranslator
        result = GoogleTranslator(source="ar", target="en").translate(text)
        return result or text
    except ImportError:
        print("  [!] deep-translator not installed. Run: pip install deep-translator")
        return text
    except Exception as e:
        print(f"  [!] Translation failed: {e}")
        return text


# ── 3. RECORD AUDIO (PUSH-TO-TALK) ────────────────────────────────────────────

def record_audio():
    r = sr.Recognizer()
    r.energy_threshold = 200
    r.dynamic_energy_threshold = True
    r.pause_threshold = 1.5

    try:
        with sr.Microphone() as source:
            print("Adjusting for ambient noise...")
            r.adjust_for_ambient_noise(source, duration=1.5)
            print("Recording... Speak now!")
            try:
                audio = r.listen(source, timeout=15, phrase_time_limit=60)
            except sr.WaitTimeoutError:
                print("No speech detected — timeout.")
                return None
    except KeyboardInterrupt:
        print("\nRecording cancelled.")
        return None

    print("Recording finished.")
    file_name = _out("raw_audio.wav")
    with open(file_name, "wb") as f:
        f.write(audio.get_wav_data())
    print(f"Raw audio saved: {file_name}")
    return file_name


# ── 4. CLEAN AUDIO (NOISE REDUCTION) ─────────────────────────────────────────

def clean_audio(input_file, output_file=None):
    if output_file is None:
        output_file = _out("clean_audio.wav")

    y, sr_rate = librosa.load(input_file, sr=None)
    print("Reducing noise...")

    # Pass 1 — stationary noise (AC hum, fan)
    reduced = nr.reduce_noise(y=y, sr=sr_rate, stationary=True, prop_decrease=0.85)
    # Pass 2 — non-stationary noise (shuffling, sudden sounds)
    reduced = nr.reduce_noise(y=reduced, sr=sr_rate, stationary=False, prop_decrease=0.75)
    # Pre-emphasis — sharpens consonants (lfilter: no extra array allocation)
    reduced = lfilter([1.0, -0.97], [1.0], reduced).astype(reduced.dtype)
    # Normalise to 95% peak
    max_val = np.max(np.abs(reduced))
    if max_val > 0:
        reduced = reduced / max_val * 0.95

    sf.write(output_file, reduced, sr_rate, subtype="PCM_16")
    print(f"Clean audio saved: {output_file}")
    return output_file


# ── 5. SPEECH → TEXT + AUTO-TRANSLATE ────────────────────────────────────────

def _transcribe_google(audio_file: str) -> str:
    """Try English first, then Arabic dialects as fallback."""
    r = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = r.record(source)

    dialects = ["en-US", PRIMARY_DIALECT] + FALLBACK_DIALECTS
    for dialect in dialects:
        try:
            print(f"  Trying {dialect}...")
            text = r.recognize_google(audio, language=dialect)
            if text:
                print(f"  Got result ({dialect}).")
                return text
        except sr.UnknownValueError:
            continue
        except sr.RequestError as e:
            return f"ERROR: API error: {e}"

    return "ERROR: Could not understand audio"


def _transcribe_whisper(audio_file: str) -> str:
    """Whisper with task=translate always outputs English regardless of input language."""
    try:
        import whisper
    except ImportError:
        return "ERROR: Whisper not installed. Run: pip install openai-whisper"

    print(f"Loading Whisper model ({WHISPER_MODEL})...")
    model = whisper.load_model(WHISPER_MODEL)
    print("Transcribing and translating to English (Whisper)...")
    result = model.transcribe(audio_file, task="translate")
    return result.get("text", "").strip()


def speech_to_text(audio_file: str):
    """
    Returns (display_text, original_text, was_translated).
    Arabic input is automatically translated to English.
    English input is returned unchanged.
    """
    print("Transcribing audio...")

    if USE_WHISPER:
        text = _transcribe_whisper(audio_file)
        return text, text, False

    raw = _transcribe_google(audio_file)

    if raw.startswith("ERROR:"):
        return raw, raw, False

    if _is_arabic(raw):
        print(f"  Arabic detected — translating to English...")
        translated = _translate_to_english(raw)
        return translated, raw, True

    return raw, raw, False


# ── 6. INTONATION HEURISTIC ──────────────────────────────────────────────────

def _ends_with_rising_pitch(audio_file: str) -> bool:
    """Returns True if RMS energy rises at end — rough signal for a question."""
    try:
        y, sr_rate = librosa.load(audio_file, sr=None, mono=True)
        half = int(0.5 * sr_rate)
        if len(y) < 2 * half:
            return False
        rms_end = np.sqrt(np.mean(y[-half:] ** 2))
        rms_pre = np.sqrt(np.mean(y[-2 * half:-half] ** 2))
        return rms_end > rms_pre * 1.15
    except Exception:
        return False


# ── 7. SAVE TO JSON ───────────────────────────────────────────────────────────

def save_to_json(sentences, filename=None):
    """
    Appends each sentence as a flat entry: {"sentence": "...", "is_question": bool}
    """
    if filename is None:
        filename = _out("nlp_log.json")

    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []

    for item in sentences:
        data.append({
            "sentence":   item["sentence"],
            "is_question": item["is_question"],
        })

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"Saved {len(sentences)} sentence(s) to: {filename}")


# ── 8. FULL PIPELINE ──────────────────────────────────────────────────────────

def push_to_talk_pipeline():
    # Step 1: Record
    raw_file = record_audio()
    if raw_file is None:
        return []

    # Step 2: Clean audio
    clean_file = clean_audio(raw_file)

    # Step 3: Transcribe + auto-translate if Arabic
    display_text, original_text, was_translated = speech_to_text(clean_file)

    # Step 4: Show output
    print("\n" + "=" * 50)
    if was_translated:
        print(f"ORIGINAL  (Arabic)  : {original_text}")
        print(f"TRANSLATED (English): {display_text}")
    else:
        print(f"TEXT: {display_text}")

    rising = _ends_with_rising_pitch(clean_file)
    if rising:
        print("Intonation hint: rising pitch at end — possible question")
    print("=" * 50)

    if not display_text or "ERROR:" in display_text:
        print("Audio unclear or error — skipped.")
        return []

    # Step 5: Detect questions on the English text
    sentences = clean_text(display_text)

    # Integrate intonation: promote last sentence if rising pitch
    if rising and sentences and not sentences[-1]["is_question"]:
        last = sentences[-1]
        last["is_question"] = True
        last["sentence"] = last["sentence"].rstrip(".!") + "?"
        last["intonation_promoted"] = True
        print("Intonation promoted last sentence to question.")

    print("\nDetected sentences:")
    for i, item in enumerate(sentences, 1):
        promoted = " [pitch]" if item.get("intonation_promoted") else ""
        label = "QUESTION" if item["is_question"] else "Statement"
        print(f"  [{i}] {label}{promoted}: {item['sentence']}")
    print("=" * 50 + "\n")

    # Step 6: Save
    if sentences:
        save_to_json(sentences)
    else:
        print("Nothing to save.")

    return sentences


# ── 9. RUN SYSTEM ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    engine = (
        f"Whisper ({WHISPER_MODEL}) [auto-translate]"
        if USE_WHISPER
        else f"Google Speech + deep-translator"
    )
    print("\n--- Smart Classroom Speech Module ---")
    print(f"    Engine   : {engine}")
    print( "    Arabic   : auto-translated to English")
    print( "    English  : kept as-is")
    print( "    Press Ctrl+C or type Q to exit\n")

    try:
        while True:
            choice = input("[ENTER] Start recording   [Q] Quit : ")
            if choice.strip().lower() == "q":
                print("Exiting...")
                break
            push_to_talk_pipeline()
    except KeyboardInterrupt:
        print("\nStopped (Ctrl+C). Exiting...")
