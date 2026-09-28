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
from dataclasses import dataclass
from src.local_model import LocalReviewer, LocalVerdict

CONFIDENCE_THRESHOLD = 0.75  # below this, always escalate regardless of verdict


@dataclass
class ReviewResult:
    verdict: str
    cwe: str
    explanation: str
    source: str  # "local" | "gemini" -- lets you track escalation rate


def escalate_to_gemini(code: str) -> ReviewResult:
    """Ask the bigger model when the local model isn't confident enough."""
    import google.generativeai as genai

    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction="You are a senior code security reviewer.",
    )

    prompt = (
        "Review the following code for security vulnerabilities and bugs. "
        'Respond with ONLY a JSON object: {"verdict": "vulnerable"|"safe", '
        '"cwe": "...", "explanation": "..."}\n\n' + code
    )
    response = model.generate_content(prompt)

    import json
    cleaned = response.text.strip().strip("`").removeprefix("json").strip()
    data = json.loads(cleaned)

    return ReviewResult(
        verdict=data.get("verdict", "unknown"),
        cwe=data.get("cwe", "unknown"),
        explanation=data.get("explanation", ""),
        source="gemini",
    )


def review_code(code: str, local_reviewer: LocalReviewer) -> ReviewResult:
    """Main router entry point -- call this from Stage 1's PR review loop."""
    local: LocalVerdict = local_reviewer.review(code)

    should_escalate = (
        local.confidence < CONFIDENCE_THRESHOLD
        or local.verdict == "vulnerable"   # always double-check positive findings --
                                            # false positives are expensive in reviewer trust
        or local.verdict == "unknown"      # local model failed to produce valid output
    )

    if not should_escalate:
        return ReviewResult(
            verdict=local.verdict,
            cwe=local.cwe,
            explanation=local.explanation,
            source="local",
        )

    return escalate_to_gemini(code)
