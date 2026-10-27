import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient

from main import app
from src.emotion_intent import EMOTIONS, INTENTS, analyze_emotion_intent

client = TestClient(app)


def test_labels_complete():
    r = analyze_emotion_intent("I am happy.")
    assert {e["label"] for e in r["emotions"]} == set(EMOTIONS)
    assert {i["label"] for i in r["intents"]} == set(INTENTS)
    assert len(EMOTIONS) == 8 and len(INTENTS) == 6


@pytest.mark.parametrize("text,emotion", [
    ("I am so happy and thrilled, this is wonderful!", "joy"),
    ("I feel so sad and lonely, I miss her and I'm heartbroken.", "sadness"),
    ("This is unacceptable, I am furious and fed up!", "anger"),
    ("I'm terrified and anxious, I'm afraid something bad will happen.", "fear"),
    ("Wow, that was completely unexpected, I'm shocked!", "surprise"),
    ("That is disgusting, absolutely revolting and nasty.", "disgust"),
    ("She is honest, reliable and trustworthy; I trust her.", "trust"),
    ("I can't wait, I'm eagerly looking forward to tomorrow.", "anticipation"),
])
def test_primary_emotion(text, emotion):
    assert analyze_emotion_intent(text)["primary_emotion"] == emotion


@pytest.mark.parametrize("text,intent", [
    ("What time does the store open? Do you know?", "inquiry"),
    ("Please send me the invoice and reset my password.", "request"),
    ("The product is broken and I am disappointed, still waiting for a refund.", "complaint"),
    ("Thank you, great job, excellent work. Highly recommend!", "praise"),
    ("I suggest you add a dark mode, it would be better if the search improved.", "suggestion"),
    ("How much is the price? I want to buy and order two.", "purchase"),
])
def test_primary_intent(text, intent):
    assert analyze_emotion_intent(text)["primary_intent"] == intent


def test_negation_flips_emotion():
    assert analyze_emotion_intent("I am not happy at all")["primary_emotion"] != "joy"


def test_scores_normalised():
    r = analyze_emotion_intent("I love this but I'm also worried. Can you help?")
    assert abs(sum(e["score"] for e in r["emotions"]) - 1.0) < 0.01
    assert abs(sum(i["score"] for i in r["intents"]) - 1.0) < 0.01


def test_neutral_text():
    r = analyze_emotion_intent("The table is made of wood.")
    assert r["primary_emotion"] == "neutral"
    assert r["primary_intent"] == "statement"
    assert r["intensity"] == "none"


def test_api_ok():
    r = client.post("/api/emotion-intent", json={"text": "I'm furious, please refund me!"})
    assert r.status_code == 200
    assert r.json()["primary_emotion"] == "anger"


def test_api_empty_rejected():
    assert client.post("/api/emotion-intent", json={"text": "   "}).status_code == 400


def test_api_batch_and_meta():
    r = client.post("/api/emotion-intent/batch", json={"texts": ["I'm happy", "I'm sad"]})
    assert r.status_code == 200 and r.json()["count"] == 2
    assert client.get("/api/labels").json()["emotions"] == EMOTIONS
    assert client.get("/api/health").json()["status"] == "ok"
    assert client.get("/").status_code == 200
