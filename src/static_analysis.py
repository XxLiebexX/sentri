"""
Stage 1: Static Analysis
Runs Semgrep against changed files before the LLM ever sees them. Cheap,
deterministic, catches known patterns fast -- its output gets fed into the
LLM reviewer as extra signal, not replaced by it.

Requires Semgrep installed: pip install semgrep
"""

import json
import subprocess
from dataclasses import dataclass


@dataclass
class StaticFinding:
    file: str
    line: int
    severity: str      # "ERROR" | "WARNING" | "INFO"
    rule_id: str
    message: str


def run_semgrep(file_paths: list[str], repo_path: str = ".") -> list[StaticFinding]:
    """
    Run Semgrep's default security/correctness ruleset against the given
    files only (not the whole repo -- keeps this fast on every PR).
    """
    if not file_paths:
        return []

    import sys
    result = subprocess.run(
        [
            "semgrep", "--config=auto", "--json", "--quiet",
            *[f"{repo_path.rstrip('/')}/{f}" for f in file_paths],
        ],
        capture_output=True, text=True, timeout=120,
    )

    if not result.stdout:
        # Semgrep exits non-zero on findings, that's expected -- only
        # treat truly empty output as a real failure.
        if result.returncode not in (0, 1):
            print(f"[static_analysis] semgrep error: {result.stderr[:300]}")
        return []

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        print("[static_analysis] Could not parse semgrep output")
        return []

    findings = []
    for r in data.get("results", []):
        findings.append(StaticFinding(
            file=r.get("path", "").replace(f"{repo_path.rstrip('/')}/", ""),
            line=r.get("start", {}).get("line", 0),
            severity=r.get("extra", {}).get("severity", "INFO"),
            rule_id=r.get("check_id", "unknown"),
            message=r.get("extra", {}).get("message", ""),
        ))

    return findings


def findings_for_chunk(chunk, all_findings: list[StaticFinding]) -> list[StaticFinding]:
    """Filter to just the findings that overlap a given CodeChunk's line range."""
    return [
        f for f in all_findings
        if f.file == chunk.file and chunk.line_start <= f.line <= chunk.line_end
    ]


def format_findings_for_prompt(findings: list[StaticFinding]) -> str:
    """Turn findings into a compact block to inject into the LLM prompt."""
    if not findings:
        return "No static analysis findings for this section."
    lines = [f"- [{f.severity}] {f.rule_id}: {f.message} (line {f.line})" for f in findings]
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    files = sys.argv[1:] or ["src/router.py"]
    findings = run_semgrep(files)
    for f in findings:
        print(f"{f.file}:{f.line} [{f.severity}] {f.rule_id} -- {f.message}")
    if not findings:
        print("No findings (or semgrep not installed -- run: pip install semgrep)")