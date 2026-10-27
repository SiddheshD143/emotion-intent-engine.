# 🎭 Emotion + Intent Engine

A FastAPI web app that reads any text and tells you **how the author feels** and **what they want**.

- **8 emotions** (Plutchik wheel): joy, sadness, anger, fear, surprise, disgust, trust, anticipation
- **6 intents**: inquiry, request, complaint, praise, suggestion, purchase
- Confidence scores for every label, an overall intensity level, and a plain-English summary
- Understands negation ("not happy"), intensifiers ("extremely angry"), emojis, `!!!` and ALL-CAPS
- Runs offline with no model downloads; optionally blends in a HuggingFace emotion model

## 🚀 Run

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Open http://localhost:8000 (UI) or http://localhost:8000/docs (Swagger). On Windows you can just double-click `start.bat`.

### Docker
```bash
docker compose up --build
```

### Optional transformer blend
Uncomment `transformers` and `torch` in `requirements.txt`, then:
```bash
USE_TRANSFORMER=1 uvicorn main:app --port 8000
```
If the model can't load, the engine falls back to the rule system automatically.

## 📡 API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/emotion-intent` | Analyse one text |
| POST | `/api/emotion-intent/batch` | Analyse up to 50 texts |
| GET | `/api/labels` | Supported emotions and intents |
| GET | `/api/health` | Health check |

```bash
curl -X POST localhost:8000/api/emotion-intent \
  -H "Content-Type: application/json" \
  -d '{"text":"I am extremely angry! My order never arrived, please refund me."}'
```

## 🗂️ Structure

```
main.py                 FastAPI app
src/emotion_intent.py   The engine
frontend/index.html     Web UI
tests/test_emotion_intent.py
```

## 🧪 Tests
```bash
python -m pytest tests/ -v
```
