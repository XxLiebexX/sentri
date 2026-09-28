"""
Stage 1: GitHub Bot
Receives a webhook when a PR opens/updates, runs the full review pipeline
(diff -> static analysis -> LLM review), and posts findings back as real
PR comments via the GitHub API.

Setup:
  1. Create a GitHub App or personal access token with repo + PR write scope
  2. Set env vars: GITHUB_TOKEN, GITHUB_WEBHOOK_SECRET
  3. Deploy this somewhere reachable (or use ngrok for local testing) and
     register the URL as your repo's webhook (Settings -> Webhooks),
     listening for "pull_request" events

Run: uvicorn src.github_bot:app --port 8001
"""

import os
import hmac
import hashlib
import tempfile
import subprocess

import requests
from fastapi import FastAPI, Request, HTTPException

from src.diff_parser import build_chunks
from src.static_analysis import run_semgrep
from src.llm_reviewer import review_all_chunks

app = FastAPI(title="Sentri GitHub Bot")

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
WEBHOOK_SECRET = os.environ.get("GITHUB_WEBHOOK_SECRET", "")
GITHUB_API = "https://api.github.com"


def verify_signature(payload_body: bytes, signature_header: str) -> bool:
    """GitHub signs webhook payloads with HMAC-SHA256 -- verify before trusting."""
    if not WEBHOOK_SECRET:
        return True  # dev mode -- set GITHUB_WEBHOOK_SECRET before going live
    if not signature_header:
        return False
    expected = "sha256=" + hmac.new(
        WEBHOOK_SECRET.encode(), payload_body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature_header)


def clone_pr_branch(clone_url: str, branch: str) -> str:
    """Shallow-clone just the PR's branch into a temp dir."""
    dest = tempfile.mkdtemp(prefix="sentri_pr_")
    subprocess.run(
        ["git", "clone", "--depth", "2", "--branch", branch, clone_url, dest],
        capture_output=True, text=True, timeout=120, check=True,
    )
    return dest


def get_changed_files(repo_path: str) -> list[str]:
    result = subprocess.run(
        ["git", "-C", repo_path, "diff", "--name-only", "HEAD~1"],
        capture_output=True, text=True,
    )
    return [f.strip() for f in result.stdout.splitlines() if f.strip()]


def post_review_comment(owner: str, repo: str, pr_number: int, commit_sha: str, finding) -> None:
    """Post a single finding as an inline PR review comment."""
    url = f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{pr_number}/comments"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }
    body = f"**[{finding.severity.upper()}] {finding.category}** (via {finding.source})\n\n{finding.message}"
    payload = {
        "body": body,
        "commit_id": commit_sha,
        "path": finding.file,
        "line": finding.line,
        "side": "RIGHT",
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=15)
    if resp.status_code >= 300:
        print(f"[github_bot] Failed to post comment on {finding.file}:{finding.line} -- {resp.text[:300]}")


@app.post("/webhook")
async def handle_webhook(request: Request):
    body = await request.body()
    signature = request.headers.get("X-Hub-Signature-256", "")

    if not verify_signature(body, signature):
        raise HTTPException(401, "Invalid webhook signature")

    event = request.headers.get("X-GitHub-Event", "")
    payload = await request.json()

    if event != "pull_request" or payload.get("action") not in ("opened", "synchronize"):
        return {"status": "ignored", "event": event}

    pr = payload["pull_request"]
    owner = payload["repository"]["owner"]["login"]
    repo = payload["repository"]["name"]
    pr_number = pr["number"]
    branch = pr["head"]["ref"]
    commit_sha = pr["head"]["sha"]
    clone_url = payload["repository"]["clone_url"]

    print(f"[github_bot] Reviewing PR #{pr_number} on {owner}/{repo} ({branch})")

    repo_path = clone_pr_branch(clone_url, branch)
    try:
        changed_files = get_changed_files(repo_path)
        if not changed_files:
            return {"status": "no_changes"}

        diff_result = subprocess.run(
            ["git", "-C", repo_path, "diff", "HEAD~1", "--unified=3"],
            capture_output=True, text=True,
        )
        chunks = build_chunks(diff_result.stdout, repo_path=repo_path)
        static_findings = run_semgrep(changed_files, repo_path=repo_path)
        findings = review_all_chunks(chunks, static_findings)

        for finding in findings:
            post_review_comment(owner, repo, pr_number, commit_sha, finding)

        print(f"[github_bot] Posted {len(findings)} findings on PR #{pr_number}")
        return {"status": "reviewed", "findings_count": len(findings)}

    finally:
        subprocess.run(["rm", "-rf", repo_path], capture_output=True)


@app.get("/health")
def health():
    return {"status": "ok", "github_token_set": bool(GITHUB_TOKEN)}
