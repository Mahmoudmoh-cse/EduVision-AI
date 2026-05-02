import re


_D2_OPENER = re.compile(
    r"(?:^|\.\s+|\?\s+|!\s+)"
    r"(what|why|how|when|where|who|which|whose|whom"
    r"|can|could|would|should|shall|will|may|might|must"
    r"|is|are|was|were|do|does|did|have|has|had|am)\b",
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

_SPLIT_MARKERS = re.compile(
    r"\s+(?="
    r"(?:but (?:what|why|how|when|where|who|can|could|do|does|did|is|are|was|were|will|would|have|has))"
    r"|(?:so (?:what|why|how|when|where|who|can|could|do|does|did|is|are|will|would))"
    r"|(?:and (?:what|why|how|when|where|who|can|could|do|does|did|is|are|will|would))"
    r"|(?:now (?:what|why|how|when|where|who|can|could|do|does|did|is|are|will|would))"
    r"|(?:like (?:what|why|how|when|where|who))"
    r"|(?:i mean (?:what|why|how|when|where|who))"
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


def is_question(text: str) -> bool:
    text = text.strip()
    if not text:
        return False

    if _GREETINGS.match(text):
        return False

    return any(
        (
            "?" in text,
            _D2_OPENER.search(text),
            _D3_EMBEDDED.search(text),
            _D4_TAG.search(text),
            _D5_INDIRECT.search(text),
        )
    )


def split_into_sentences(text: str) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []

    if any(punctuation in text for punctuation in [".", "?", "!"]):
        parts = re.split(r"(?<=[.?!])\s+", text)
        return [
            sentence
            for part in parts
            for sentence in _split_by_embedded_questions(part)
            if sentence
        ]

    parts = re.split(
        r"\s+(?:and then|but then|however|also|because|after that"
        r"|okay so|alright so|now|anyway)\s+",
        text,
        flags=re.IGNORECASE,
    )

    return [
        sentence
        for part in parts
        for sentence in _split_by_embedded_questions(part)
        if sentence
    ]


def clean_text(text: str) -> list[dict[str, object]]:
    results = []
    for sentence in split_into_sentences(text):
        question = is_question(sentence)
        results.append(
            {
                "sentence": _format_sentence(sentence, question),
                "is_question": question,
            }
        )
    return results


def _split_by_embedded_questions(text: str) -> list[str]:
    parts = _SPLIT_MARKERS.split(text)
    if len(parts) == 1:
        parts = _QUESTION_STARTERS.split(text)

    return [part.strip() for part in parts if part and part.strip()]


def _format_sentence(sentence: str, question: bool) -> str:
    sentence = sentence.strip().rstrip(".?!")
    if not sentence:
        return sentence

    sentence = sentence[0].upper() + sentence[1:]
    return sentence + ("?" if question else ".")
