"""
Emotion + Intent Engine — FastAPI backend
=========================================
Detects 8 emotions and 6 intents in text with confidence scores.

Run:  uvicorn main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

import os
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.emotion_intent import EMOTIONS, INTENTS, analyze_emotion_intent

app = FastAPI(
    title="Emotion + Intent Engine",
    description="🎭 Detect 8 emotions and 6 intents in any text, with confidence scores.",
    version="1.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


class TextRequest(BaseModel):
    text: str = Field(..., examples=["I'm so frustrated! My order never arrived, please refund me."])


class BatchRequest(BaseModel):
    texts: List[str] = Field(..., max_length=50)


@app.get("/", include_in_schema=False)
def root():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.post("/api/emotion-intent", tags=["Emotion & Intent"])
def emotion_intent(req: TextRequest):
    """Analyse one text: 8 emotions + 6 intents with confidence scores."""
    if not req.text.strip():
        raise HTTPException(400, "Text cannot be empty.")
    if len(req.text) > 10_000:
        raise HTTPException(400, "Text too long (max 10,000 characters).")
    return analyze_emotion_intent(req.text)


@app.post("/api/emotion-intent/batch", tags=["Emotion & Intent"])
def emotion_intent_batch(req: BatchRequest):
    """Analyse up to 50 texts at once."""
    texts = [t for t in req.texts if t.strip()]
    if not texts:
        raise HTTPException(400, "Provide at least one non-empty text.")
    return {"count": len(texts), "results": [analyze_emotion_intent(t) for t in texts]}


@app.get("/api/labels", tags=["Meta"])
def labels():
    """List the supported emotion and intent labels."""
    return {"emotions": EMOTIONS, "intents": INTENTS}


@app.get("/api/health", tags=["Meta"])
def health():
    return {"status": "ok", "platform": "Emotion + Intent Engine v1.0"}
