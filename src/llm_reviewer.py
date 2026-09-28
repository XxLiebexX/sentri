"""
Stage 1: LLM Reviewer
The core of Stage 1. Takes a CodeChunk (diff + full context + optional
Stage 0 spec context) plus static analysis findings, and produces a
structured Finding by calling the local model + Gemini router built
earlier. This is what turns "a diff" into "an actual review."
"""

from dataclasses import dataclass
from src.diff_parser import CodeChunk
from src.static_analysis import StaticFinding, findings_for_chunk, format_findings_for_prompt
from src.router import review_code, ReviewResult
from src.local_model import LocalReviewer


@dataclass
class Finding:
    file: str
    line: int
    severity: str          # "error" | "warning" | "info"
    category: str          # e.g. CWE id, or "style" / "logic"
    message: str
    source: str            # "local" | "gemini" | "static" -- powers the dashboard comparison


def build_review_prompt(chunk: CodeChunk, static_findings: list[StaticFinding]) -> str:
    """
    Assemble the full context the model needs: the diff, the full
    surrounding function, relevant project conventions from Stage 0,
    and any static analysis hits already found for this section.
    """
    static_block = format_findings_for_prompt(static_findings)

    spec_block = chunk.spec_context or "No project spec context available."

    return f"""Review this code change.

PROJECT CONTEXT:
{spec_block}

STATIC ANALYSIS FINDINGS FOR THIS SECTION:
{static_block}

DIFF:
{chunk.diff_text}

FULL FUNCTION CONTEXT:
{chunk.full_context}

Judge whether this change introduces a bug, security vulnerability, or
violates the project's own conventions shown above. Static analysis
findings are a hint, not the final word -- confirm, dismiss, or add to
them based on real understanding of this code."""


def review_chunk(
    chunk: CodeChunk,
    local_reviewer: LocalReviewer,
    all_static_findings: list[StaticFinding],
) -> list[Finding]:
    """
    Review a single CodeChunk end-to-end: gather static findings for it,
    build the prompt, route through local model / Gemini, return
    normalized Finding objects ready to post as PR comments.
    """
    static_hits = findings_for_chunk(chunk, all_static_findings)
    prompt = build_review_prompt(chunk, static_hits)

    result: ReviewResult = review_code(prompt, local_reviewer)

    findings = []

    # LLM verdict becomes a Finding
    if result.verdict == "vulnerable":
        findings.append(Finding(
            file=chunk.file,
            line=chunk.line_start,
            severity="error",
            category=result.cwe,
            message=result.explanation,
            source=result.source,
        ))

    # Static findings that the LLM didn't already cover get surfaced too --
    # cheap deterministic hits shouldn't get lost just because the LLM
    # verdict was "safe" on the chunk overall.
    for sf in static_hits:
        findings.append(Finding(
            file=sf.file,
            line=sf.line,
            severity=sf.severity.lower(),
            category=sf.rule_id,
            message=sf.message,
            source="static",
        ))

    return findings


def review_all_chunks(
    chunks: list[CodeChunk],
    all_static_findings: list[StaticFinding],
) -> list[Finding]:
    """Review every changed chunk in a PR. Loads the local model once, reuses it."""
    local_reviewer = LocalReviewer()
    all_findings = []
    for chunk in chunks:
        all_findings.extend(review_chunk(chunk, local_reviewer, all_static_findings))
    return all_findings
