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
from src.scan_repo import scan_repo
from src.local_model import LocalReviewer

app = FastAPI(title="Sentri API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this before deploying anywhere real
    allow_methods=["*"],
    allow_headers=["*"],
)

import asyncio

_model_lock = asyncio.Lock()
_local_reviewer: LocalReviewer | None = None
_model_load_task = None

async def _load_model_bg():
    global _local_reviewer
    async with _model_lock:
        if _local_reviewer is not None:
            return
        try:
            _local_reviewer = await asyncio.to_thread(LocalReviewer)
            print("[startup] Local model loaded successfully.")
        except Exception as e:
            print(f"[startup] Could not load local model yet: {e}")
            _local_reviewer = None

@app.on_event("startup")
def load_model():
    global _model_load_task
    _model_load_task = asyncio.create_task(_load_model_bg())

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
async def review(req: ReviewRequest):
    if not req.code.strip():
        raise HTTPException(400, "No code provided")

    if _local_reviewer is None:
        if _model_load_task and not _model_load_task.done():
            raise HTTPException(503, "Local model is currently loading, please try again in a few seconds.")
        raise HTTPException(
            503,
            "Local model not loaded yet -- run finetune/train.py first, "
            "or check backend startup logs.",
        )

    start = time.time()
    print("[api] Starting local review...")
    
    local = await asyncio.to_thread(_local_reviewer.review, req.code)
    print(f"[api] Local review finished. Verdict: {local.verdict}")
    
    escalated = (
        local.confidence < CONFIDENCE_THRESHOLD
        or local.verdict in ("vulnerable", "unknown")
    )

    print("[api] Calling Gemini...")
    gemini_result = await escalate_to_gemini(req.code)
    print("[api] Gemini review finished.")

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


class ScanRequest(BaseModel):
    repo_url: str
    max_files: int = 5


class RepoFindingOut(BaseModel):
    file: str
    line: int
    severity: str
    category: str
    message: str
    source: str


class ScanResponse(BaseModel):
    repo_url: str
    project_type: str
    purpose: str
    files_scanned: int
    files_skipped: int
    findings: list[RepoFindingOut]


@app.post("/api/scan-repo", response_model=ScanResponse)
async def scan_repo_endpoint(req: ScanRequest):
    if not req.repo_url.strip():
        raise HTTPException(400, "No repo URL provided")

    try:
        report = await scan_repo(req.repo_url, max_files=req.max_files, skip_ai=False, skip_spec=False, local_reviewer=_local_reviewer)
    except Exception as e:
        raise HTTPException(500, f"Scan failed: {str(e)}")

    return ScanResponse(
        repo_url=report.repo_url,
        project_type=report.project_spec.get("project_type", "unknown"),
        purpose=report.project_spec.get("purpose", ""),
        files_scanned=report.files_scanned,
        files_skipped=report.files_skipped,
        findings=[
            RepoFindingOut(file=f.file, line=f.line, severity=f.severity, category=f.category, message=f.message, source=f.source)
            for f in report.findings
        ],
    )


class OverviewRequest(BaseModel):
    repo_url: str

@app.post("/api/overview")
async def get_project_overview(req: OverviewRequest):
    if not req.repo_url.strip():
        raise HTTPException(400, "No repo URL provided")
        
    from src.ingest import ingest_repo
    from src.summarize import summarize_repo
    import asyncio
    
    try:
        print(f"[api] Fetching overview for {req.repo_url}")
        # Run the cloning and summarization in a thread to prevent blocking
        snapshot = await asyncio.to_thread(ingest_repo, req.repo_url)
        spec = await asyncio.to_thread(summarize_repo, snapshot)
        
        return {
            "repo_url": spec.repo_url,
            "commit_hash": spec.commit_hash,
            "project_type": spec.project_type,
            "purpose": spec.purpose,
            "architecture": spec.architecture,
            "languages": snapshot.languages,
            "total_files": snapshot.total_files,
            "key_modules": spec.key_modules,
            "conventions": spec.conventions,
            "risk_areas": spec.risk_areas
        }
    except Exception as e:
        raise HTTPException(500, f"Overview generation failed: {str(e)}")


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": _local_reviewer is not None}
