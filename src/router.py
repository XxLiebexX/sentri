"""
The core "cheap filter, expensive escalation" pattern:

  1. Run the fine-tuned local model first (fast, free, no API cost)
  2. If it's confident AND the verdict is "safe" -> done, no escalation needed
  3. If it's low-confidence, flags "vulnerable", or fails to parse ->
     escalate to Gemini for a second, more capable opinion

This keeps API costs down (most code in a PR is fine and gets filtered
out cheaply) while still catching hard cases with the stronger model.
"""

import os
import time
from dataclasses import dataclass
from src.local_model import LocalReviewer, LocalVerdict

CONFIDENCE_THRESHOLD = 0.75


@dataclass
class ReviewResult:
    verdict: str
    cwe: str
    explanation: str
    source: str


def escalate_to_gemini(code: str, _retries: int = 4) -> ReviewResult:
    import google.generativeai as genai
    from google.api_core.exceptions import ResourceExhausted

    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    model = genai.GenerativeModel(
        model_name="gemini-flash-lite-latest",
        system_instruction="You are a senior code security reviewer analyzing code for defensive purposes, to help developers fix vulnerabilities.",
    )

    prompt = (
        "Review the following code for security vulnerabilities and bugs, for defensive code review purposes. "
        "Respond with ONLY a JSON object: {\"verdict\": \"vulnerable\"|\"safe\", "
        "\"cwe\": \"...\", \"explanation\": \"...\"}\n\n" + code
    )

    response = None
    for attempt in range(_retries):
        try:
            response = model.generate_content(prompt)
            break
        except ResourceExhausted:
            if attempt == _retries - 1:
                raise
            wait = 20
            print(f"[router] Rate limited, waiting {wait}s (attempt {attempt+1}/{_retries})...")
            time.sleep(wait)

    # Handle blocked / empty responses instead of crashing on response.text
    if not response or not response.candidates:
        reason = getattr(getattr(response, "prompt_feedback", None), "block_reason", "unknown")
        print(f"[router] Gemini blocked this request (reason: {reason}) -- falling back to local-only result")
        return ReviewResult(
            verdict="unknown",
            cwe="unknown",
            explanation=f"Gemini blocked this request (reason: {reason}). Treat as needing manual review.",
            source="gemini_blocked",
        )

    import json
    try:
        cleaned = response.text.strip().strip("`").removeprefix("json").strip()
        data = json.loads(cleaned)
    except Exception as e:
        print(f"[router] Could not parse Gemini response: {e}")
        return ReviewResult(
            verdict="unknown",
            cwe="unknown",
            explanation="Could not parse Gemini response.",
            source="gemini_blocked",
        )

    return ReviewResult(
        verdict=data.get("verdict", "unknown"),
        cwe=data.get("cwe", "unknown"),
        explanation=data.get("explanation", ""),
        source="gemini",
    )


def review_code(code: str, local_reviewer: LocalReviewer) -> ReviewResult:
    local: LocalVerdict = local_reviewer.review(code)

    should_escalate = (
        local.confidence < CONFIDENCE_THRESHOLD
        or local.verdict == "vulnerable"
        or local.verdict == "unknown"
    )

    if not should_escalate:
        return ReviewResult(
            verdict=local.verdict,
            cwe=local.cwe,
            explanation=local.explanation,
            source="local",
        )

    return escalate_to_gemini(code)
