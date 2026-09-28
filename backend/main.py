"""
Sentri backend API.

Exposes the local model + Gemini router as an HTTP endpoint the frontend
calls. Also returns the local-only and Gemini-only verdicts separately
(not just the final routed one) so the UI can show a side-by-side
comparison -- that transparency is the whole point of the dashboard.

Run: uvicorn backend.main:app --reload --port 8000
"""

import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.router import escalate_to_gemini, CONFIDENCE_THRESHOLD
from src.local_model import LocalReviewer

app = FastAPI(title="Sentri API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this before deploying anywhere real
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loaded once at startup -- loading the model per-request would be very slow.
_local_reviewer: LocalReviewer | None = None


@app.on_event("startup")
def load_model():
    global _local_reviewer
    try:
        _local_reviewer = LocalReviewer()
    except Exception as e:
        # Lets the API still boot (e.g. for frontend dev) even before
        # you've trained a model yet -- reviews will just report the error.
        print(f"[startup] Could not load local model yet: {e}")
        _local_reviewer = None


class ReviewRequest(BaseModel):
    code: str


class VerdictOut(BaseModel):
    verdict: str
    cwe: str
    explanation: str
    confidence: float | None = None


class ReviewResponse(BaseModel):
    local: VerdictOut
    gemini: VerdictOut
    final_source: str          # "local" | "gemini" -- which one the router trusted
    escalated: bool
    confidence_threshold: float
    latency_ms: int


@app.post("/api/review", response_model=ReviewResponse)
def review(req: ReviewRequest):
    if not req.code.strip():
        raise HTTPException(400, "No code provided")

    if _local_reviewer is None:
        raise HTTPException(
            503,
            "Local model not loaded yet -- run finetune/train.py first, "
            "or check backend startup logs.",
        )

    start = time.time()

    local = _local_reviewer.review(req.code)
    escalated = (
        local.confidence < CONFIDENCE_THRESHOLD
        or local.verdict in ("vulnerable", "unknown")
    )

    # Always also call Gemini for the comparison view, even when the router
    # wouldn't have escalated -- this is what makes the dashboard genuinely
    # useful for evaluating the local model's quality, not just running it.
    gemini_result = escalate_to_gemini(req.code)

    latency_ms = int((time.time() - start) * 1000)

    return ReviewResponse(
        local=VerdictOut(
            verdict=local.verdict,
            cwe=local.cwe,
            explanation=local.explanation,
            confidence=round(local.confidence, 3),
        ),
        gemini=VerdictOut(
            verdict=gemini_result.verdict,
            cwe=gemini_result.cwe,
            explanation=gemini_result.explanation,
        ),
        final_source="gemini" if escalated else "local",
        escalated=escalated,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        latency_ms=latency_ms,
    )


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": _local_reviewer is not None}
