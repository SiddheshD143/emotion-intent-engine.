"""
Emotion + Intent Engine
=======================
Detects 8 emotions (Plutchik wheel) and 6 intents in a piece of text and
returns normalised confidence scores for each.

Emotions : joy, sadness, anger, fear, surprise, disgust, trust, anticipation
Intents  : inquiry, request, complaint, praise, suggestion, purchase

The default engine is a fast, dependency-free lexicon + rule system that
understands negation ("not happy"), intensifiers ("very angry"), emojis,
exclamation marks and ALL-CAPS emphasis.

Optionally, set USE_TRANSFORMER=1 to blend in a HuggingFace emotion model
(j-hartmann/emotion-english-distilroberta-base). If the model cannot be
loaded the engine silently keeps using the rule system.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import Dict, List

EMOTIONS = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "trust", "anticipation"]
INTENTS = ["inquiry", "request", "complaint", "praise", "suggestion", "purchase"]

EMOTION_EMOJI = {
    "joy": "😊", "sadness": "😢", "anger": "😠", "fear": "😨",
    "surprise": "😲", "disgust": "🤢", "trust": "🤝", "anticipation": "🤩",
}
INTENT_EMOJI = {
    "inquiry": "❓", "request": "🙏", "complaint": "😤",
    "praise": "👏", "suggestion": "💡", "purchase": "🛒",
}

# ── Emotion lexicon ───────────────────────────────────────────────────────────
_EMOTION_WORDS: Dict[str, List[str]] = {
    "joy": """happy happiness glad delighted delight joy joyful cheerful thrilled excited
        wonderful fantastic amazing great awesome love loved loving lovely enjoy enjoyed
        enjoying pleased pleasure fun celebrate celebrating blessed grateful thankful
        thanks brilliant perfect excellent superb beautiful smile laugh yay hooray""",
    "sadness": """sad unhappy depressed miserable heartbroken sorrow grief cry crying cried
        lonely alone hopeless disappointed disappointing upset gloomy down hurt pain
        painful miss missing lost loss regret sorry tears mourning devastated""",
    "anger": """angry anger furious mad rage outraged annoyed annoying irritated irritating
        hate hated hating terrible awful worst horrible unacceptable ridiculous stupid
        idiot idiots fed useless pathetic infuriating frustrated frustrating frustration
        livid fuming scam fraud""",
    "fear": """afraid scared fear frightened terrified worried worry anxious anxiety nervous
        panic dread threat danger dangerous unsafe horror alarmed uneasy concerned
        concern risk risky trembling""",
    "surprise": """surprised surprise shocked shock unexpected suddenly sudden wow whoa omg
        astonished astonishing unbelievable incredible stunned amazed speechless
        unexpectedly""",
    "disgust": """disgust disgusting disgusted gross nasty revolting vile sick sickening
        filthy dirty repulsive yuck eww rotten stinks smelly appalling""",
    "trust": """trust trusted trustworthy reliable dependable honest loyal faithful confident
        confidence safe secure support supportive believe recommend recommended
        authentic genuine credible respect respected""",
    "anticipation": """looking forward excited eager anticipate anticipating expect expecting
        hope hoping can't wait soon upcoming tomorrow plan planning ready soon
        countdown waiting await awaiting""",
}

_EMOJI_EMOTION = {
    "😀": "joy", "😃": "joy", "😄": "joy", "😁": "joy", "😊": "joy", "😍": "joy", "🥰": "joy", "🎉": "joy",
    "😢": "sadness", "😭": "sadness", "😞": "sadness", "💔": "sadness", "☹": "sadness",
    "😠": "anger", "😡": "anger", "🤬": "anger", "👎": "anger",
    "😨": "fear", "😱": "fear", "😰": "fear", "😟": "fear",
    "😲": "surprise", "😮": "surprise", "🤯": "surprise",
    "🤢": "disgust", "🤮": "disgust", "😖": "disgust",
    "🤝": "trust", "🙏": "trust", "👍": "trust",
    "🤩": "anticipation", "⏳": "anticipation", "🔜": "anticipation",
}

_NEGATIONS = {"not", "no", "never", "n't", "cannot", "cant", "dont", "doesnt", "didnt",
              "isnt", "wasnt", "wont", "wouldnt", "hardly", "barely", "without"}
_INTENSIFIERS = {"very": 1.5, "really": 1.4, "extremely": 1.8, "so": 1.3, "totally": 1.5,
                 "absolutely": 1.7, "incredibly": 1.7, "super": 1.4, "utterly": 1.7,
                 "completely": 1.6, "deeply": 1.5, "too": 1.2}
_DIMINISHERS = {"slightly": 0.6, "somewhat": 0.7, "bit": 0.7, "little": 0.7, "kinda": 0.7}

# When an emotion is negated, it flips towards its opposite.
_OPPOSITE = {
    "joy": "sadness", "sadness": "joy", "anger": "trust", "fear": "trust",
    "surprise": None, "disgust": "joy", "trust": "anger", "anticipation": None,
}

# ── Intent patterns ───────────────────────────────────────────────────────────
_INTENT_PATTERNS: Dict[str, List[tuple]] = {
    "inquiry": [
        (r"\?", 2.0),
        (r"^\s*(what|when|where|who|whom|whose|why|how|which|is|are|do|does|did|can|could|will|would|should)\b", 1.8),
        (r"\b(wondering|curious|want to know|need to know|any idea|do you know|tell me|let me know|information about|details about|explain)\b", 2.0),
    ],
    "request": [
        (r"\b(please|kindly|pls|plz)\b", 1.5),
        (r"\b(can|could|would|will) you\b", 1.5),
        (r"\b(i need|i want|i require|i'd like|i would like|need you to|help me|assist|send me|give me|provide|arrange|schedule|book|cancel|refund|reset|update|change|fix)\b", 2.0),
    ],
    "complaint": [
        (r"\b(complain|complaint|issue|problem|broken|defective|damaged|faulty|not working|doesn't work|does not work|didn't work|stopped working|failed|failure|error|bug|crash|crashed)\b", 2.0),
        (r"\b(disappointed|unacceptable|terrible|awful|worst|horrible|useless|waste|rude|late|delay|delayed|overcharged|scam|fraud|refund)\b", 1.8),
        (r"\b(never again|fed up|sick of|tired of|still waiting|no response|no one|nobody)\b", 2.0),
    ],
    "praise": [
        (r"\b(great|excellent|amazing|awesome|fantastic|wonderful|brilliant|perfect|superb|outstanding|love|loved|best)\b", 1.8),
        (r"\b(thank you|thanks|thankful|grateful|appreciate|appreciated|well done|good job|great job|kudos|impressed|impressive)\b", 2.2),
        (r"\b(highly recommend|recommended|five stars|5 stars|satisfied|happy with)\b", 2.0),
    ],
    "suggestion": [
        (r"\b(suggest|suggestion|recommend|propose|idea|advice|advise|improve|improvement|enhance|consider|maybe|perhaps)\b", 1.8),
        (r"\b(should|could be|would be better|would be nice|would be great|it'd be|it would help|why not|how about|what if|better if|wish)\b", 2.0),
        (r"\b(add|include|implement|introduce|feature request)\b", 1.2),
    ],
    "purchase": [
        (r"\b(buy|buying|purchase|purchasing|order|ordering|checkout|add to cart|price|pricing|cost|discount|offer|deal|coupon|subscribe|subscription|upgrade|plan|quote|invoice|payment|pay)\b", 2.0),
        (r"\b(interested in|looking to buy|want to buy|how much|in stock|available|delivery|shipping|ship)\b", 2.0),
    ],
}

_TOKEN_RE = re.compile(r"[a-z']+")

_LEXICON: Dict[str, set] = {}
for _emo, _blob in _EMOTION_WORDS.items():
    _LEXICON[_emo] = set(_blob.split())
# multi-word phrases need separate handling
_PHRASES = {
    "looking forward": "anticipation", "can't wait": "anticipation", "cant wait": "anticipation",
    "fed up": "anger", "sick of": "disgust", "freaked out": "fear",
    "thank you": "joy", "well done": "joy", "good job": "joy",
}


# ── Optional transformer ──────────────────────────────────────────────────────
_PIPE = None
_PIPE_TRIED = False
_HF_MAP = {"joy": "joy", "sadness": "sadness", "anger": "anger", "fear": "fear",
           "surprise": "surprise", "disgust": "disgust"}


def _get_pipeline():
    global _PIPE, _PIPE_TRIED
    if _PIPE_TRIED:
        return _PIPE
    _PIPE_TRIED = True
    if os.getenv("USE_TRANSFORMER", "0") != "1":
        return None
    try:
        from transformers import pipeline  # type: ignore
        _PIPE = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None,
        )
    except Exception:
        _PIPE = None
    return _PIPE


def _transformer_scores(text: str) -> Dict[str, float]:
    pipe = _get_pipeline()
    if pipe is None:
        return {}
    try:
        out = pipe(text[:1500])
        out = out[0] if out and isinstance(out[0], list) else out
        return {_HF_MAP[o["label"]]: float(o["score"]) for o in out if o["label"] in _HF_MAP}
    except Exception:
        return {}


# ── Emotion scoring ───────────────────────────────────────────────────────────
def _emotion_raw(text: str) -> Dict[str, float]:
    lower = text.lower().replace("’", "'")
    tokens = _TOKEN_RE.findall(lower)
    scores: Dict[str, float] = defaultdict(float)

    for phrase, emo in _PHRASES.items():
        scores[emo] += 1.5 * lower.count(phrase)

    for i, tok in enumerate(tokens):
        for emo, words in _LEXICON.items():
            if tok not in words:
                continue
            weight = 1.0
            negated = False
            for prev in tokens[max(0, i - 3):i]:
                if prev in _NEGATIONS or prev.endswith("n't"):
                    negated = True
                weight *= _INTENSIFIERS.get(prev, 1.0) * _DIMINISHERS.get(prev, 1.0)
            if negated:
                opp = _OPPOSITE.get(emo)
                if opp:
                    scores[opp] += weight * 0.7
            else:
                scores[emo] += weight

    for ch in text:
        emo = _EMOJI_EMOTION.get(ch)
        if emo:
            scores[emo] += 1.5

    # Emphasis: "!!!" and ALL-CAPS words amplify the strongest emotions
    emphasis = 1.0 + min(text.count("!"), 4) * 0.08
    caps_words = [w for w in re.findall(r"\b[A-Z]{3,}\b", text) if w not in {"USA", "FAQ", "CEO", "API"}]
    emphasis += min(len(caps_words), 3) * 0.1
    return {k: v * emphasis for k, v in scores.items()}


def _normalise(raw: Dict[str, float], keys: List[str]) -> Dict[str, float]:
    total = sum(raw.get(k, 0.0) for k in keys)
    if total <= 0:
        return {k: 0.0 for k in keys}
    return {k: round(raw.get(k, 0.0) / total, 4) for k in keys}


def detect_emotions(text: str) -> Dict[str, float]:
    raw = _emotion_raw(text)
    rule = _normalise(raw, EMOTIONS)
    tr = _transformer_scores(text)
    if tr and sum(rule.values()) > 0:
        blended = {k: 0.5 * rule[k] + 0.5 * tr.get(k, 0.0) for k in EMOTIONS}
        return _normalise(blended, EMOTIONS)
    if tr:
        blended = {k: tr.get(k, 0.0) for k in EMOTIONS}
        return _normalise(blended, EMOTIONS)
    return rule


# ── Intent scoring ────────────────────────────────────────────────────────────
def detect_intents(text: str) -> Dict[str, float]:
    lower = text.lower().replace("’", "'")
    raw: Dict[str, float] = defaultdict(float)
    for intent, patterns in _INTENT_PATTERNS.items():
        for pattern, weight in patterns:
            hits = len(re.findall(pattern, lower, flags=re.MULTILINE))
            if hits:
                raw[intent] += weight * min(hits, 3)
    return _normalise(raw, INTENTS)


# ── Public API ────────────────────────────────────────────────────────────────
def _ranked(scores: Dict[str, float], emoji: Dict[str, str]) -> List[dict]:
    return [
        {"label": k, "emoji": emoji[k], "score": v, "percent": round(v * 100, 1)}
        for k, v in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    ]


def _intensity(text: str, emotions: Dict[str, float]) -> str:
    if max(emotions.values(), default=0) == 0:
        return "none"
    bangs = text.count("!")
    caps = len(re.findall(r"\b[A-Z]{3,}\b", text))
    boosters = sum(1 for t in _TOKEN_RE.findall(text.lower()) if t in _INTENSIFIERS)
    level = bangs + caps * 2 + boosters * 2
    return "high" if level >= 5 else "medium" if level >= 2 else "low"


def _summary(primary_emo: str, primary_int: str, emo_conf: float, int_conf: float) -> str:
    if emo_conf == 0 and int_conf == 0:
        return "No strong emotional or intent signals were detected — the text reads as neutral."
    parts = []
    if emo_conf > 0:
        parts.append(f"The dominant emotion is **{primary_emo}** ({emo_conf:.0%})")
    if int_conf > 0:
        parts.append(f"the likely intent is **{primary_int}** ({int_conf:.0%})")
    else:
        parts.append("no clear intent was found")
    return ", and ".join(parts) + "."


def analyze_emotion_intent(text: str) -> dict:
    """Return emotion + intent scores, rankings, intensity and a summary."""
    text = text.strip()
    emotions = detect_emotions(text)
    intents = detect_intents(text)

    emo_ranked = _ranked(emotions, EMOTION_EMOJI)
    int_ranked = _ranked(intents, INTENT_EMOJI)
    top_emo = emo_ranked[0] if emo_ranked[0]["score"] > 0 else None
    top_int = int_ranked[0] if int_ranked[0]["score"] > 0 else None

    primary_emotion = top_emo["label"] if top_emo else "neutral"
    primary_intent = top_int["label"] if top_int else "statement"
    return {
        "text_length": len(text),
        "word_count": len(text.split()),
        "primary_emotion": primary_emotion,
        "primary_emotion_emoji": top_emo["emoji"] if top_emo else "😐",
        "primary_intent": primary_intent,
        "primary_intent_emoji": top_int["emoji"] if top_int else "💬",
        "emotion_confidence": top_emo["score"] if top_emo else 0.0,
        "intent_confidence": top_int["score"] if top_int else 0.0,
        "intensity": _intensity(text, emotions),
        "emotions": emo_ranked,
        "intents": int_ranked,
        "summary": _summary(primary_emotion, primary_intent,
                            top_emo["score"] if top_emo else 0.0,
                            top_int["score"] if top_int else 0.0),
        "engine": "rules+transformer" if _PIPE is not None else "rules",
    }
